import abc
import csv
import sys
from functools import reduce
from operator import and_
from os.path import join, isfile

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ParameterSampler
from sklearn.utils import shuffle

from config import DIR_PATH
from helpers import show_progress


class Evaluator(abc.ABC):
    """Constant class-level properties; same for all inheriting classes"""
    test_data_dir = join(DIR_PATH, 'test_data')
    api_gender_key_name = 'gender'  # change this in inheriting class if API response denotes the gender differently

    @property
    @abc.abstractmethod
    def uses_full_name(self):
        """Returns Boolean whether API can be requested using full name string."""
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def gender_evaluator(self):
        """Name string of the service. Used for names of files with evaluation results."""
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def gender_response_mapping(self):
        """Mapping of gender assignments from the service such as 'male' or 'female' to 'm' and 'f'."""
        return {}

    @property
    @abc.abstractmethod
    def tuning_params(self):
        """Attributes in the API response that can be used for model tuning, e.g. 'probability' or 'count'"""
        return ()

    def __init__(self, data_source):
        self.file_path_raw_data = join(self.test_data_dir, 'raw_data', data_source + '.csv')
        self.file_path_evaluated_data = join(self.test_data_dir, self.gender_evaluator,
                                             data_source + '_' + self.gender_evaluator + '.csv')
        self.test_data = pd.DataFrame()
        self.api_response = []
        self.api_call_completed = False
        self.confusion_matrix = None

    def load_data(self, evaluated=False, return_frame=False):
        """Load data with names and gender assignments from a CSV file.
        Data is either stored in the attribute 'test_data' or returned.

        :param evaluated: set to 'False' if you want to load data that has not been evaluated yet through
        a service; set to 'True' if you want to load data with gender assignments from a service (Boolean)
        :param return_frame: set to 'False' to load data into attribute 'test_data' (default);
        'True' to get the data returned (Boolean)
        :return: pandas DataFrame
        """
        from_file = self.file_path_raw_data if not evaluated else self.file_path_evaluated_data
        try:
            test_data = pd.read_csv(from_file, keep_default_na=False)
            expected_columns = ['first_name', 'middle_name', 'last_name', 'full_name', 'gender']
            if sum([item in test_data.columns for item in expected_columns]) == len(expected_columns):
                test_data[expected_columns] = test_data[expected_columns].fillna('')

                if return_frame:
                    return test_data
                else:
                    self.test_data = test_data
            else:
                print("Some expected columns are missing; data not loaded.")

        except FileNotFoundError:
            print("File not found")

    """Methods for fetching gender from a service and storing them in a file or in an attribute"""

    def fetch_gender(self, save_to_dump=True):
        """Fetch gender predictions from dump file at path 'file_path_evaluated_data' if present or from API if not.
        """
        if isfile(self.file_path_evaluated_data):
            self.load_data(evaluated=True)
            print('Reading data from dump file {}'.format(self.file_path_evaluated_data))
        else:
            self._fetch_gender_from_api()
            self._extend_test_data_by_api_response()
            if self.api_call_completed:
                self._translate_api_response()
                if save_to_dump:
                    print('Saving data to dump file {}'.format(self.file_path_evaluated_data))
                    self.dump_evaluated_test_data_to_file()
            else:
                print('API call did not complete. Check error and try again.')

    def update_selected_records(self, index):
        """Calls API on the selected records and updates the corresponding rows
        :param index: sub-index of 'test_data' of the selected records (pandas Index)
        """
        for ind in index:
            row = self.test_data.loc[ind]
            print('Updating entry {}'.format(ind))
            print('''Calling API for name:\nfirst_name: {}\tmiddle_name: {}\t \
                  last_name: {}\tfull_name: {}'''.format(row.first_name, row.middle_name,
                                                         row.last_name, row.full_name))
            api_resp = self._process_row_for_api_call(row)
            api_resp = {'api_{}'.format(k): v for (k, v) in api_resp.items()}
            api_resp['gender_infered'] = self.gender_response_mapping[api_resp['api_gender']]
            for (k, v) in api_resp.items():
                self.test_data.loc[ind, k] = v
            for k in self.test_data.columns[:5]:
                self.test_data.loc[ind, k] = row[k]
        print('Data updated in dump file {}'.format(self.file_path_evaluated_data))
        self.dump_evaluated_test_data_to_file()

    def _fetch_gender_from_api(self):
        """Fetches gender assignments of names in 'test_data' from an API or Python module
        and appends them to the list attribute 'api_response'.
        If 'api_response' is not empty, i.e. gender assignments have already been retrieved for a subset of 'test_data',
        then the method starts with the row index corresponding to new names."""

        print('Fetching gender data from API of service {}'.format(self.gender_evaluator))
        start_position = len(self.api_response)
        print('Starting from record: {}'.format(start_position))

        for i, row in enumerate(self.test_data[start_position:].itertuples()):
            show_progress(i)
            try:
                api_resp = self._process_row_for_api_call(row)
                if api_resp:
                    self.api_response.append(api_resp)
                else:
                    # If api_resp is None it means that the evaluation failed -> break the loop
                    break
            except Exception as e:
                # This prints any unforeseen error when processing each row
                # Should never reach here b/c every class should handle its own potential errors when
                # calling their APIs
                print('An unexpected error occured')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(exc_type, exc_tb.tb_lineno)
                print(e)
                break

    @classmethod
    def _process_row_for_api_call(cls, row):
        """Take a row from 'test_data' and process it to for the api call.
        Which cells of the row will be used and how they are processed depends on the class attribute 'uses_full_name',
        and whether a middle name exists.
        :param row: row from the attribute 'test_data' (pandas Series)
        :return: dictionary with the API response if the call has succeeded, None otherwise
        """

        first, mid, last, full = row.first_name, row.middle_name, row.last_name, row.full_name

        if cls.uses_full_name is True:
            api_resp = cls._fetch_gender_with_full_name(full)
        else:
            if mid == '':
                api_resp = cls._fetch_gender_with_first_last(first, last)
            else:
                api_resp = cls._fetch_gender_with_first_mid_last(first, mid, last)
        return api_resp

    @classmethod
    @abc.abstractmethod
    def _fetch_gender_with_full_name(cls, full):
        """Calls the API with full name for services that accept full name."""

    @classmethod
    @abc.abstractmethod
    def _fetch_gender_with_first_last(cls, first, last):
        """Decides how to handle the API call when only a first and last name are present."""

    @classmethod
    @abc.abstractmethod
    def _fetch_gender_with_first_mid_last(cls, first, mid, last):
        """Decides how to handle the API call when a first, middle, and last name are present."""

    @staticmethod
    @abc.abstractmethod
    def _call_api(name):
        """Sends a request with one or more names to an API and returns a response."""

    def dump_evaluated_test_data_to_file(self):
        if 'gender_infered' in self.test_data.columns:
            self.test_data.to_csv(self.file_path_evaluated_data, index=False, quoting=csv.QUOTE_NONNUMERIC)
        else:
            print("Test data has not been evaluated yet, won't dump")

    def _extend_test_data_by_api_response(self):
        """Add response from service to self.test_data if number of responses equals number of rows in self.test_data.
        Hereby rename the column with gender assignment from service to 'api_gender'."""
        if len(self.api_response) == len(self.test_data):
            api_response = pd.DataFrame(self.api_response).add_prefix('api_')
            self.test_data = pd.concat([self.test_data, api_response], axis=1)
            self.api_call_completed = True
            print('... API calls completed')
        else:
            print("\nResponse from API contains less results than request. Try again?")
            self.api_call_completed = False

    def _translate_api_response(self, **kwargs):
        """Create new column 'gender_infered' in 'test_data' by translating values in the column
        'api_gender' to 'f' and 'm' using the class attribute 'gender_response_mapping'.
        If keyword arguments of the form 'param_name=param_threshold' are provided then 'gender_infered' is set to
        'u' if the value in column 'param_name' is less than 'param_threshold'.
        Otherwise, all values other than 'm' and 'f' are set to 'u'.
        """
        self.test_data['gender_infered'] = self.test_data['api_gender']
        self.test_data.replace({'gender_infered': self.gender_response_mapping}, inplace=True)
        self.test_data.loc[~self.test_data['gender_infered'].isin(['f', 'm']), 'gender_infered'] = 'u'

        if kwargs != {}:
            # connect all expressions in kwargs with Boolean 'and'
            and_mask = reduce(and_,
                              [(self.test_data[str(param_name)] < param_threshold) for param_name, param_threshold in
                               kwargs.items()])
            self.test_data.loc[and_mask, 'gender_infered'] = 'u'

    """Methods for exploration of service responses and error metrics"""

    def compare_ground_truth_with_inference(self, gender, gender_infered):
        """Displays rows of 'test_data' where value in column 'gender_infered'  differs from that in 'gender'."""
        return self.test_data[
            (self.test_data.gender == gender) & (self.test_data.gender_infered == gender_infered)]

    @staticmethod
    def compute_confusion_matrix(df, col_true='gender', col_pred='gender_infered'):
        f_f = len(df[(df[col_true] == 'f') & (df[col_pred] == 'f')])
        f_m = len(df[(df[col_true] == 'f') & (df[col_pred] == 'm')])
        f_u = len(df[(df[col_true] == 'f') & (df[col_pred] == 'u')])
        m_f = len(df[(df[col_true] == 'm') & (df[col_pred] == 'f')])
        m_m = len(df[(df[col_true] == 'm') & (df[col_pred] == 'm')])
        m_u = len(df[(df[col_true] == 'm') & (df[col_pred] == 'u')])
        u_f = len(df[(df[col_true] == 'u') & (df[col_pred] == 'f')])
        u_m = len(df[(df[col_true] == 'u') & (df[col_pred] == 'm')])
        u_u = len(df[(df[col_true] == 'u') & (df[col_pred] == 'u')])

        return pd.DataFrame([[f_f, f_m, f_u], [m_f, m_m, m_u], [u_f, u_m, u_u]], index=['f', 'm', 'u'],
                            columns=['f_pred', 'm_pred', 'u_pred'])

    def set_confusion_matrix(self):
        self.confusion_matrix = self.compute_confusion_matrix(self.test_data)

    @staticmethod
    def compute_error_without_unknown(conf_matrix):
        """Corresponds to 'errorCodedWithoutNA' from genderizeR-paper:
        https://journal.r-project.org/archive/2016/RJ-2016-002/RJ-2016-002.pdf"""
        error_without_unknown = (conf_matrix.loc['f', 'm_pred'] + conf_matrix.loc['m', 'f_pred']) / \
                                (conf_matrix.loc['f', 'm_pred'] + conf_matrix.loc['m', 'f_pred'] +
                                 conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['m', 'm_pred'])

        return error_without_unknown

    @staticmethod
    def compute_error_with_unknown(conf_matrix):
        """
        Corresponds to 'errorCoded' from genderizeR-paper:
        https://journal.r-project.org/archive/2016/RJ-2016-002/RJ-2016-002.pdf"""
        true_f_and_m = conf_matrix.loc['f', :].sum() + conf_matrix.loc['m', :].sum()
        true_pred_f_and_m = conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['m', 'm_pred']
        error_with_unknown = (true_f_and_m - true_pred_f_and_m) / true_f_and_m

        return error_with_unknown

    @staticmethod
    def compute_error_unknown(conf_matrix):
        """Corresponds 'naCoded' from genderizeR-paper:
        https://journal.r-project.org/archive/2016/RJ-2016-002/RJ-2016-002.pdf"""
        true_f_and_m = conf_matrix.loc['f', :].sum() + conf_matrix.loc['m', :].sum()
        error_unknown = (conf_matrix.loc['f', 'u_pred'] + conf_matrix.loc['m', 'u_pred']) / true_f_and_m

        return error_unknown

    @staticmethod
    def compute_error_gender_bias(conf_matrix):
        """Corresponds 'errorGenderBias' from genderizeR-paper:
        https://journal.r-project.org/archive/2016/RJ-2016-002/RJ-2016-002.pdf"""
        error_gender_bias = (conf_matrix.loc['m', 'f_pred'] - conf_matrix.loc['f', 'm_pred']) / \
                            (conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['f', 'm_pred'] +
                             conf_matrix.loc['m', 'f_pred'] + conf_matrix.loc['m', 'm_pred'])

        return error_gender_bias

    @staticmethod
    def compute_weighted_error(conf_matrix, eps=0.2):
        """Compute weighted version of 'error_with_unknown', where terms related to classifying 'f' and 'm' as 'u'
        is multiplied with 'eps'."""
        numer = (conf_matrix.loc['m', 'f_pred'] + conf_matrix.loc['f', 'm_pred'] + eps * (
                conf_matrix.loc['m', 'u_pred'] + conf_matrix.loc['f', 'u_pred']))
        denom = (conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['f', 'm_pred'] + conf_matrix.loc['m', 'f_pred'] +
                 conf_matrix.loc['m', 'm_pred'] + eps * (
                         conf_matrix.loc['m', 'u_pred'] + conf_matrix.loc['f', 'u_pred']))
        return numer / denom

    def compute_all_errors(self):
        self.set_confusion_matrix()
        error_with_unknown = self.compute_error_with_unknown(self.confusion_matrix)
        error_without_unknown = self.compute_error_without_unknown(self.confusion_matrix)
        error_unknown = self.compute_error_unknown(self.confusion_matrix)
        error_gender_bias = self.compute_error_gender_bias(self.confusion_matrix)
        weighted_error = self.compute_weighted_error(self.confusion_matrix)
        return [error_with_unknown, error_without_unknown, error_gender_bias, error_unknown, weighted_error]

    """Methods for parameter tuning"""

    def compute_k_fold_cv_score(self, n_splits, param_range, error_func, constraint_func=None, constraint_val=None,
                                stratified=True, verbose=False):
        """Compute cross validation score using 'n_splits' randomly chosen train-test splits of the dataframe
        'test_data'. Remove rows for which gender is unknown since 'u' is not a real class.

        :param n_splits: number of folds; should be at least 2 (int)
        :param param_range: list of tuning parameter-value pairs used for 'training' the function (list of dict)
        :param error_func: one of the error functions in this class
        :param stratified: Boolean whether to use stratified folds
        :param constraint_func: constraint function
        :param constraint_val: maximum threshold for 'constraint_func'
        :param verbose: set to 'True' if you want to see prints (Boolean)
        :return: mean error on the test set folds (float)
        """
        train_test_splits = self.build_train_test_splits(self.test_data, n_splits=n_splits, stratified=stratified)
        nfold_errors = []  # errors on each of the k test sets for the optimal function on corresponding train set
        try:
            for train_index, test_index in train_test_splits:
                test_error, train_error, best_params = self.tune_params(param_range, error_func, train_index,
                                                                        test_index,
                                                                        constraint_func, constraint_val)
                if verbose:
                    print("minimal train error:", train_error, "corresponding test error:", test_error)
                    print("params for lowest train error:", best_params)
                nfold_errors.append(test_error)
            if verbose:
                print("Average test error:", np.mean(nfold_errors))
            return np.mean(nfold_errors)
        except:
            print("No parameter values satisfied given constraint")

    def tune_params(self, param_grid, error_func, train_index, test_index, constraint_func=None, constraint_val=None):
        """Find tuple of parameter values from 'param_grid' that minimize an error on a training set and return corresponding
        parameter values and errors on train and test set
        :param param_grid: list of tuning parameter-value pairs (list of dicts)
        :param error_func: one of the error functions in this class
        :param train_index: sub-index of attribute 'test_data' which defines the training set
        :param test_index: sub-index of attribute 'test_data' which defines the test set
        :param constraint_func: constraint function
        :param constraint_val: maximum threshold for 'constraint_func'
        :return: error on test, error on training set, parameter values from the grid which minimizes error function
        on the training set

        Example (instance 'evaluator' with 'test_data' consisting of 5 rows):
        random_list = np.random.rand(len(evaluator.test_data)) < 0.8
        train_index = evaluator.test_data.index[random_list]
        test_index = evaluator.test_data.index[~random_list]
        param_grid = [{'api_count': 10, 'api_probability': 0.7}, {'api_count': 10, 'api_probability': 0.9},
                      {'api_count': 100, 'api_probability': 0.7}, {'api_count': 100, 'api_probability': 0.9}]
        test_error, train_error, best_params = evaluator.tune_params(param_grid, evaluator.compute_error_without_unknown,
                                                                    train_index, test_index)
        >>> 0.0397683397683, 0.0434445306439, {'api_probability': 0.5, 'api_count': 50}
        """
        param_to_error_mapping = self.compute_train_test_error_for_param_range(param_grid, error_func, train_index,
                                                                               test_index)
        if constraint_func:
            # compute constraint error func on test set and restrict to only those param values for which the constraint error is less than 'constraint_val'
            param_to_constraint_mapping = self.compute_error_for_param_range(param_grid, constraint_func, test_index)
            param_to_error_mapping = {k: v for k, v in param_to_error_mapping.items() if
                                      param_to_constraint_mapping[k] < constraint_val}

        param_to_error_mapping = sorted(param_to_error_mapping.items(),
                                        key=lambda x: x[1][0])  # sort by lowest training error
        try:
            best_param_values_and_errors = param_to_error_mapping[0]
            min_train_error = best_param_values_and_errors[1][0]
            min_test_error = best_param_values_and_errors[1][1]
            param_min_train_error = dict(zip(self.tuning_params, best_param_values_and_errors[0]))

            return min_test_error, min_train_error, param_min_train_error
        except IndexError:
            print("No parameter values satisfied given constraint")
            return 1, None, None  # Error 1 is higher than any value expected

    @abc.abstractmethod
    def preprocess_tuning_params(self):
        pass

    def shuffle_data(self):
        self.test_data = shuffle(self.test_data, random_state=1).reset_index()

    def sample_parameters(self, n_iter=20, method='square', random_state=None):
        """Draw 'n_iter' random samples from the distribution of 'tuning_params'.
        :param n_iter: integer (default=20)
        :param method: set to 'square' if you want 'n-iter' values per dimension of the parameter space;
        set to 'constant' if no dependence on dimension (string).
        :param random_state: None or integer
        :return: dictionary where keys are names of tuning parameters and values are randomly drawn parameter values.
        """
        tuning_params = self.tuning_params
        param_name_to_param_value = {}

        for param_name in tuning_params:
            param_name_to_param_value[param_name] = self.test_data[param_name].values

        if method == 'square':
            n_param_values = n_iter ** len(tuning_params)
        else:
            n_param_values = n_iter

        param_list = list(ParameterSampler(param_name_to_param_value, n_iter=n_param_values, random_state=random_state))
        zero_el = {k: 0 for k in param_list[0].keys()}  # corresponds to the default response of API

        param_list = [dict(t) for t in set([tuple(d.items()) for d in param_list])] + [
            zero_el]  # deduplicate and add zero element
        param_name_to_param_value = [dict((k, round(v, 6)) for (k, v) in d.items()) for d in param_list]

        return param_name_to_param_value

    def remove_rows_with_unknown_gender(self, gender=True, gender_infered=False):
        if gender:
            self.test_data = self.test_data[self.test_data.gender != 'u']
        if gender_infered:
            self.test_data = self.test_data[
                (self.test_data.gender_infered == 'f') | (self.test_data.gender_infered == 'm')]
        self.test_data.reset_index(inplace=True)

    @staticmethod
    def build_train_test_splits(df, n_splits, stratified=False):
        y = df['gender']

        if stratified is False:
            kf = KFold(n_splits=n_splits, random_state=1, shuffle=False)
            return list(kf.split(df))
        else:
            skf = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=False)
            return list(skf.split(df, y))

    def compute_error_for_param_range(self, param_range, error_func, index):
        param_to_error_mapping = {}
        for param_values in param_range:
            self._translate_api_response(**param_values)
            conf_matrix = self.compute_confusion_matrix(self.test_data.loc[index, :])
            error = error_func(conf_matrix)
            tuning_param_values = tuple(param_values[param] for param in
                                        self.tuning_params)  # keep only parameter values and get rid of their names
            param_to_error_mapping[tuning_param_values] = error
        return param_to_error_mapping

    def compute_train_test_error_for_param_range(self, param_range, error_func, train_index, test_index):
        """Compute error on train and test set for certain choice of tuning parameters.
        :param param_range: list of tuning parameter-value pairs (list of dict)
        :param error_func: one of the error functions in this class
        :param train_index: sub-index of attribute 'test_data' which defines the training set
        :param test_index: sub-index of attribute 'test_data' which defines the test set
        :return: error on training and test set for specified 'param_values' (tuple of floats)

        Example (instance 'evaluator'):
        random_list = np.random.rand(len(evaluator.test_data)) < 0.8
        train_index = evaluator.test_data.index[random_list]
        test_index = evaluator.test_data.index[~random_list]
        evaluator.compute_train_test_error_for_param_range([
        {'api_count': 1, 'api_probability': 0.5},
        {'api_count': 1, 'api_probability': 0.6},
        {'api_count': 10, 'api_probability': 0.5},
        {'api_count': 10, 'api_probability': 0.6}],
        evaluator.compute_error_unknown, train_index, test_index)
        >>> {(1, 0.5): (0.097258485639686684, 0.097152428810720268),
        (1, 0.6): (0.097258485639686684, 0.097152428810720268),
        (10, 0.5): (0.097258485639686684, 0.097152428810720268),
        (10, 0.6): (0.10204525674499565, 0.098827470686767172)}
        """
        param_to_error_mapping = {}
        for param_values in param_range:
            self._translate_api_response(**param_values)
            # print(self.test_data.gender_infered.value_counts())
            conf_matrix_train = self.compute_confusion_matrix(self.test_data.loc[train_index, :])
            conf_matrix_test = self.compute_confusion_matrix(self.test_data.loc[test_index, :])
            error_train = error_func(conf_matrix_train)
            error_test = error_func(conf_matrix_test)
            tuning_param_values = tuple(param_values[param] for param in self.tuning_params)

            # error_train, error_test = self.compute_train_test_error(param_values, error_func, train_index, test_index)
            param_to_error_mapping[tuning_param_values] = (error_train, error_test)
        return param_to_error_mapping
