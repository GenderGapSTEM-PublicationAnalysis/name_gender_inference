# TODO: document methods
import pandas as pd
import os
import csv


class Evaluator(object):
    """Class-level properties"""
    gender_evaluator = None

    def __init__(self, file_path):
        self.file_path = file_path
        self.test_data = pd.DataFrame()
        self.api_response = None
        self.is_test_data_schema_correct = None
        self.confusion_matrix = None
        self.error_without_unknown = None
        self.error_with_unknown = None
        self.error_unknown = None
        self.error_gender_bias = None

    def load_data(self):
        try:
            test_data = pd.read_csv(self.file_path)
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
        # TODO: make file path creation prettier
        # Decide that evaluation exists if column gender_infered is in test_data
        if 'gender_infered' in self.test_data.columns:
            filename, extension = os.path.splitext(self.file_path)
            dump_file = filename + '_' + self.gender_evaluator + extension
            self.test_data.to_csv(dump_file, index=False,
                                  quoting=csv.QUOTE_NONNUMERIC)
        else:
            print("Test data has not been evaluated yet, won't dump")

    def compare_ground_truth_with_inference(self, true_gender, gender_infered):
        """'true_gender' and 'infered_gender' should be one of the strings 'u', 'm', 'f'.
        Displays rows of 'test_data' where inference differed from ground truth."""
        return self.test_data[
            (self.test_data.gender == true_gender) & (self.test_data.gender_infered == gender_infered)]

    def fetch_gender(self, save_to_dump=True):
        # TODO: fetching from API needs a caching decorator in order to be more efficient with quotas
        """Fetches gender predictions, either from dump if present or from API if not
        It relies on the dump file having a particular naming convention consistent with 
        self.dump_test_data_with_gender_inference_to_file"""
        if self.gender_evaluator is None:
            raise ValueError("Missing gender_evaluator needed to fetch the gender")
        # Name the dump file like the original but adding a _genderevaluator qualifier
        # This works with all extensions, but later we sort of assume that the file is .csv 
        filename, extension = os.path.splitext(self.file_path)
        dump_file = filename + '_' + self.gender_evaluator + extension
        # Try opening the dump file, else resort to calling the API
        try:
            # TODO: replace by load method above
            self.test_data = pd.read_csv(dump_file)
            print('Reading data from dump file {}'.format(dump_file))
        except FileNotFoundError:
            print('Fetching gender data from API of service {}'.format(self.gender_evaluator))
            self._fetch_gender_from_api()
            if save_to_dump:
                print('Saving data to dump file {}'.format(dump_file))
                self.dump_test_data_with_gender_inference_to_file()

    def extend_test_data_by_api_response(self, api_response, gender_mapping):
        api_response = pd.DataFrame(api_response).rename(columns={"gender": "gender_infered"})
        if len(api_response) == len(self.test_data):
            self.test_data = pd.concat([self.test_data, api_response], axis=1)
        else:
            print("response from API contains less results than request. Try again?")

        self.test_data.replace(to_replace={"gender_infered": gender_mapping}, inplace=True)

    def _fetch_gender_from_api(self):
        """Method is implemented in the classes inheriting from this. """
        # TODO: check if this needs a change.
        pass

    def compute_confusion_matrix(self):
        if self.gender_evaluator is not None:
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
