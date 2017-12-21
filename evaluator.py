# TODO: document methods
import pandas as pd
import abc
import csv


class Evaluator(abc.ABC):
    """Constant class-level properties; same for all inheriting classes"""
    raw_data_prefix = 'test_data/raw_data/test_data_'
    data_suffix = '.csv'
    api_gender_key_name = 'gender'  # change this in inheriting class if API response denotes the gender differently

    @property
    @abc.abstractmethod
    def gender_evaluator(self):
        """Name string of the service. Used for names of files with evaluation results."""
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def gender_response_mapping(self):
        """mapping of gender assignments from the service to 'm', 'f' and 'u'"""
        return 'Should never reach here'

    def __init__(self, data_source):
        self.data_source = data_source
        self.file_path_raw_data = self.raw_data_prefix + self.data_source + self.data_suffix
        self.file_path_evaluated_data = 'test_data/' + self.gender_evaluator + '/test_data_' + \
                                        self.data_source + '_' + self.gender_evaluator + self.data_suffix
        self.test_data = pd.DataFrame()
        self.api_response = []
        self.is_test_data_schema_correct = None
        self.confusion_matrix = None
        self.error_without_unknown = None
        self.error_with_unknown = None
        self.error_unknown = None
        self.error_gender_bias = None

    def load_data(self):
        try:
            test_data = pd.read_csv(self.file_path_raw_data, keep_default_na=False)
            expected_columns = ['first_name', 'middle_name', 'last_name', 'full_name', 'gender']
            if sum([item in test_data.columns for item in expected_columns]) == \
                    len(expected_columns):
                self.test_data = test_data
                self.is_test_data_schema_correct = True
                for col in expected_columns:
                    self.test_data[col].fillna('', inplace=True)
            else:
                print("Some expected columns are missing; data not loaded.")

        except FileNotFoundError:
            print("File not found")

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

    def fetch_gender(self, save_to_dump=True):
        # TODO: change try-except structure to if-else since it is not real exception handling
        """Fetches gender predictions, either from dump if present or from API if not
        It relies on the dump file having a particular naming convention consistent with 
        self.dump_test_data_with_gender_inference_to_file"""
        # Try opening the dump file, else resort to calling the API
        try:
            # TODO: replace by load method above
            self.test_data = pd.read_csv(self.file_path_evaluated_data)
            print('Reading data from dump file {}'.format(self.file_path_evaluated_data))
        except FileNotFoundError:
            print('Fetching gender data from API of service {}'.format(self.gender_evaluator))
            self._fetch_gender_from_api()
            self.extend_test_data_by_api_response()
            self._translate_api_response()
            if save_to_dump:
                print('Saving data to dump file {}'.format(self.file_path_evaluated_data))
                self.dump_test_data_with_gender_inference_to_file()

    def extend_test_data_by_api_response(self):
        """Add response from service to self.test_data if number of responses equals number of rows in self.test_data.
        Hereby rename the column with gender assignment from service to 'api_gender'."""
        if len(self.api_response) == len(self.test_data):
            api_response = pd.DataFrame(self.api_response).add_prefix('api_')
            self.test_data = pd.concat([self.test_data, api_response], axis=1)
        else:
            print("Response from API contains less results than request. Try again?")

    def _translate_api_response(self):
        """Create new column 'gender_infered' in self.test_data which translates gender assignments from the
        service to 'f', 'm' and 'u'."""
        self.test_data['gender_infered'] = self.test_data['api_gender']
        self.test_data.replace({'gender_infered': self.gender_response_mapping}, inplace=True)

    @abc.abstractmethod
    def _fetch_gender_from_api(self):
        """Fetches gender assignments from an API or Python module"""

    @staticmethod
    @abc.abstractmethod
    def _call_api(name):
        """Sends a request with one or more names to an API and returns a response."""

    def compute_confusion_matrix(self):
        f_f = len(self.test_data[(self.test_data.gender == 'f') & (self.test_data.gender_infered == 'f')])
        f_m = len(self.test_data[(self.test_data.gender == 'f') & (self.test_data.gender_infered == 'm')])
        f_u = len(self.test_data[(self.test_data.gender == 'f') & (self.test_data.gender_infered == 'u')])
        m_f = len(self.test_data[(self.test_data.gender == 'm') & (self.test_data.gender_infered == 'f')])
        m_m = len(self.test_data[(self.test_data.gender == 'm') & (self.test_data.gender_infered == 'm')])
        m_u = len(self.test_data[(self.test_data.gender == 'm') & (self.test_data.gender_infered == 'u')])
        u_f = len(self.test_data[(self.test_data.gender == 'u') & (self.test_data.gender_infered == 'f')])
        u_m = len(self.test_data[(self.test_data.gender == 'u') & (self.test_data.gender_infered == 'm')])
        u_u = len(self.test_data[(self.test_data.gender == 'u') & (self.test_data.gender_infered == 'u')])

        self.confusion_matrix = pd.DataFrame([[f_f, f_m, f_u], [m_f, m_m, m_u], [u_f, u_m, u_u]],
                                             index=['f', 'm', 'u'],
                                             columns=['f_pred', 'm_pred', 'u_pred'])

    """Error metrics from paper on genderizeR; see p.26 and p.27 (Table 2) for an explanation of the errors"""

    # TODO: implement that confusion matrix needs to be filled

    def compute_error_with_unknown(self):
        """Corresponds to 'errorCoded' in genderizeR"""
        true_f_and_m = self.confusion_matrix.loc['f', :].sum() + self.confusion_matrix.loc['m', :].sum()
        true_pred_f_and_m = self.confusion_matrix.loc['f', 'f_pred'] + self.confusion_matrix.loc['m', 'm_pred']
        self.error_with_unknown = (true_f_and_m - true_pred_f_and_m) / true_pred_f_and_m

    def compute_error_without_unknown(self):
        """Corresponds 'errorCodedWithoutNA' from genderizeR"""
        self.error_without_unknown = (self.confusion_matrix.loc['f', 'm_pred'] +
                                      self.confusion_matrix.loc['m', 'f_pred']) / \
                                     (self.confusion_matrix.loc['f', 'm_pred'] + self.confusion_matrix.loc[
                                         'm', 'f_pred'] +
                                      self.confusion_matrix.loc['f', 'f_pred'] + self.confusion_matrix.loc[
                                          'm', 'm_pred'])

    def compute_error_unknown(self):
        """Corresponds 'naCoded' from genderizeR"""
        true_f_and_m = self.confusion_matrix.loc['f', :].sum() + self.confusion_matrix.loc['m', :].sum()
        self.error_unknown = (self.confusion_matrix.loc['f', 'u_pred'] +
                              self.confusion_matrix.loc['m', 'u_pred']) / true_f_and_m

    def compute_error_gender_bias(self):
        """Corresponds '' from genderizeR"""
        self.error_gender_bias = (self.confusion_matrix.loc['m', 'f_pred'] +
                                  self.confusion_matrix.loc['f', 'm_pred']) / \
                                 (self.confusion_matrix.loc['f', 'f_pred'] + self.confusion_matrix.loc['f', 'm_pred'] +
                                  self.confusion_matrix.loc['m', 'f_pred'] + self.confusion_matrix.loc['m', 'm_pred'])

    def compute_all_errors(self):
        self.compute_confusion_matrix()
        self.compute_error_with_unknown()
        self.compute_error_without_unknown()
        self.compute_error_unknown()
        self.compute_error_gender_bias()
        # print(self.confusion_matrix)
        # print("error counting prediction as 'unknown gender' as classification errors: ", self.error_with_unknown)
        # print("error ignoring prediction as 'unknown gender' : ", self.error_without_unknown)
        # print("error counting proportion of names with unpredicted gender: ", self.error_unknown)
        # print("error where negative value suggestes that more women than men are missclassified: ",
        #       self.error_gender_bias)
        return [self.error_with_unknown, self.error_without_unknown, self.error_gender_bias, self.error_unknown]
