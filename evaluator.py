# TODO: document methods
import abc
import csv
import os
import sys

import itertools
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold

from helpers import show_progress


class Evaluator(abc.ABC):
    """Constant class-level properties; same for all inheriting classes"""
    raw_data_prefix = 'test_data/raw_data/test_data_'
    data_suffix = '.csv'
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
        """Mapping of gender assignments from the service to 'm', 'f' and 'u'"""
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def tuning_params(self):
        """Attributes in the API response that can be used for model tuning, e.g. 'probability' or 'count'"""
        return ()

    def __init__(self, data_source):
        self.data_source = data_source
        self.file_path_raw_data = self.raw_data_prefix + self.data_source + self.data_suffix
        self.file_path_evaluated_data = 'test_data/' + self.gender_evaluator + '/test_data_' + \
                                        self.data_source + '_' + self.gender_evaluator + self.data_suffix
        self.test_data = pd.DataFrame()
        self.api_response = []
        self.api_call_completed = False
        self.confusion_matrix = None

    def load_data(self, evaluated=False, return_frame=False):
        from_file = self.file_path_raw_data if not evaluated else self.file_path_evaluated_data
        try:
            test_data = pd.read_csv(from_file, keep_default_na=False)
            expected_columns = ['first_name', 'middle_name', 'last_name', 'full_name', 'gender']
            if sum([item in test_data.columns for item in expected_columns]) == \
                    len(expected_columns):
                if return_frame:
                    # Call with return_frame=True to get the data returned
                    test_data[expected_columns] = test_data[expected_columns].fillna('')
                    return test_data
                else:
                    # Call with default return_frame=False to load data into attribute
                    self.test_data = test_data
                    self.test_data[expected_columns] = self.test_data[expected_columns].fillna('')
            else:
                print("Some expected columns are missing; data not loaded.")

        except FileNotFoundError:
            print("File not found")

    def fetch_gender(self, save_to_dump=True):
        """Fetches gender predictions, either from dump if present or from API if not
        It relies on the dump file having a particular naming convention consistent with 
        self.dump_test_data_with_gender_inference_to_file"""
        # Try opening the dump file, else resort to calling the API
        if os.path.isfile(self.file_path_evaluated_data):
            self.load_data(evaluated=True)
            print('Reading data from dump file {}'.format(self.file_path_evaluated_data))
        else:
            self._fetch_gender_from_api()
            self.extend_test_data_by_api_response()
            if self.api_call_completed:
                self._translate_api_response()
                if save_to_dump:
                    print('Saving data to dump file {}'.format(self.file_path_evaluated_data))
                    self.dump_test_data_with_gender_inference_to_file()
            else:
                print('API call did not complete. Check error and try again.')

    def extend_test_data_by_api_response(self):
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
        """Create new column 'gender_infered' in self.test_data which translates gender assignments from the
        service to 'f', 'm' and 'u'."""
        self.test_data['gender_infered'] = self.test_data['api_gender']
        self.test_data.replace({'gender_infered': self.gender_response_mapping}, inplace=True)

        if kwargs is not None:
            genders = [('gender_infered', 'm'), ('gender_infered', 'f')]
            thresholds = [(str(k), v) for k, v in kwargs.items()]

            for g in genders:
                for t in thresholds:
                    self.test_data.loc[
                        ((self.test_data[g[0]] == g[1]) & (self.test_data[t[0]] < t[1])), 'gender_infered'] = 'u'

    def _fetch_gender_from_api(self):
        """Fetches gender assignments from an API or Python module."""
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
        """Takes a row from the test data frame and processes it to make the relevant api call.

        Returns a dict api_resp with the data to be appended to self.api_response if the call succeded
        Else it returs None, which breaks the execution of the for loop over the rows.
        """
        # How a row will processed depends first on whether a mid name exists
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
        """Calls the API with full name, for methods that accept full name."""

    @classmethod
    @abc.abstractmethod
    def _fetch_gender_with_first_last(cls, first, last):
        """Decides how to handle the API call when a first and last name are present."""

    @classmethod
    @abc.abstractmethod
    def _fetch_gender_with_first_mid_last(cls, first, mid, last):
        """Decides how to handle the API call when a first, middle, and last name are present."""

    @staticmethod
    @abc.abstractmethod
    def _call_api(name):
        """Sends a request with one or more names to an API and returns a response."""

    def update_selected_records(self, indices):
        """Calls API on the selected records and updates the corresponding rows"""
        for ind in indices:
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
        self.dump_test_data_with_gender_inference_to_file()

    def dump_test_data_with_gender_inference_to_file(self):
        if 'gender_infered' in self.test_data.columns:
            self.test_data.to_csv(self.file_path_evaluated_data, index=False, quoting=csv.QUOTE_NONNUMERIC)
        else:
            print("Test data has not been evaluated yet, won't dump")

    def compare_ground_truth_with_inference(self, true_gender, gender_infered):
        """'true_gender' and 'infered_gender' should be one of the strings 'u', 'm', 'f'.
        Displays rows of 'test_data' where inference differed from ground truth."""
        return self.test_data[
            (self.test_data.gender == true_gender) & (self.test_data.gender_infered == gender_infered)]

    """Methods for parameter tuning"""

    @classmethod
    def build_parameter_grid(cls, *args):
        """Takes one or many lists of parameter values as args which refer to the
        tuning_params attribute of the class in the given order.
        Returns the cross-product of these values as key-value pairs.
        """
        assert len(args) == len(cls.tuning_params)
        return [dict(zip(cls.tuning_params, param_tuple)) for param_tuple in list(itertools.product(*args))]

    def remove_rows_with_unknown_gender(self):
        self.test_data = self.test_data[self.test_data.gender != 'u']
        self.test_data.reset_index(inplace=True)

    @staticmethod
    def build_train_test_splits(df, n_splits, stratified=False, shuffle=True):
        y = df['gender']

        if stratified is False:
            kf = KFold(n_splits=n_splits, random_state=1, shuffle=shuffle)
            return list(kf.split(df))
        else:
            skf = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=shuffle)
            return list(skf.split(df, y))

    # def compute_train_test_error(grid_point, train_index, test_index):
    #     evaluator._translate_api_response(**grid_point)
    #     conf_matrix_train = evaluator.compute_confusion_matrix(evaluator.test_data.loc[train_index, :])
    #     conf_matrix_test = evaluator.compute_confusion_matrix(evaluator.test_data.loc[test_index, :])
    #     error_train = evaluator.build_error_without_unknown(conf_matrix_train)
    #     error_test = evaluator.build_error_without_unknown(conf_matrix_test)
    #     print(item.values(), error_train, error_test)
    #     param_to_error_mapping[(item[param_1], item[param_2])] = (error_train, error_test)
    #     # print(param_to_error_mapping)


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
        # TODO: check whether we really need this attribute
        self.confusion_matrix = self.compute_confusion_matrix(self.test_data)

    @staticmethod
    def compute_error_without_unknown(conf_matrix):
        """Corresponds 'errorCodedWithoutNA' from genderizeR"""
        error_without_unknown = (conf_matrix.loc['f', 'm_pred'] + conf_matrix.loc['m', 'f_pred']) / \
                                (conf_matrix.loc['f', 'm_pred'] + conf_matrix.loc['m', 'f_pred'] +
                                 conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['m', 'm_pred'])

        return error_without_unknown

    """Error metrics from paper on genderizeR; see p.26 and p.27 (Table 2) for an explanation of the errors"""

    @staticmethod
    def compute_error_with_unknown(conf_matrix):
        """Corresponds to 'errorCoded' in genderizeR"""
        true_f_and_m = conf_matrix.loc['f', :].sum() + conf_matrix.loc['m', :].sum()
        true_pred_f_and_m = conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['m', 'm_pred']
        error_with_unknown = (true_f_and_m - true_pred_f_and_m) / true_pred_f_and_m

        return error_with_unknown

    @staticmethod
    def compute_error_unknown(conf_matrix):
        """Corresponds 'naCoded' from genderizeR"""
        true_f_and_m = conf_matrix.loc['f', :].sum() + conf_matrix.loc['m', :].sum()
        error_unknown = (conf_matrix.loc['f', 'u_pred'] + conf_matrix.loc['m', 'u_pred']) / true_f_and_m

        return error_unknown

    @staticmethod
    def compute_error_gender_bias(conf_matrix):
        """Corresponds '' from genderizeR"""
        error_gender_bias = (conf_matrix.loc['m', 'f_pred'] + conf_matrix.loc['f', 'm_pred']) / \
                            (conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['f', 'm_pred'] +
                             conf_matrix.loc['m', 'f_pred'] + conf_matrix.loc['m', 'm_pred'])

        return error_gender_bias

    def compute_all_errors(self):
        self.set_confusion_matrix()
        error_with_unknown = self.compute_error_with_unknown(self.confusion_matrix)
        error_without_unknown = self.compute_error_without_unknown(self.confusion_matrix)
        error_unknown = self.compute_error_unknown(self.confusion_matrix)
        error_gender_bias = self.compute_error_gender_bias(self.confusion_matrix)
        return [error_with_unknown, error_without_unknown, error_gender_bias, error_unknown]
