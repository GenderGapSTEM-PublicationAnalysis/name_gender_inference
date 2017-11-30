# TODO: document methods
import pandas as pd
from genderize import Genderize


class GenderEvaluator(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.test_data = pd.DataFrame()
        self.is_test_data_schema_correct = None
        self.gender_evaluator = None
        self.confusion_matrix = None
        self.error_without_unknown = None
        self.error_with_unknown = None
        self.error_unknown = None
        self.error_gender_bias = None

    def load_data(self):
        try:
            self.test_data = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print("File not found")

    def check_data_columns(self):
        expected_columns = ['first_name', 'middle_name', 'last_name', 'gender']
        if sum([item in self.test_data.columns for item in expected_columns]) == len(expected_columns):
            self.is_test_data_schema_correct = True

    def fetch_gender_from_genderizeio(self):
        """Fetches gender predictions from genderize.io using self.test_data.first_name
        and merges them with self.test_data."""
        # TODO: genderize.io is sensitive towards non-word characters.
        names = self.test_data.first_name.tolist()
        result = []
        i = 0
        while i < len(names):
            result.extend(Genderize().get(names[i: i + 10]))
            i += 10

        result = pd.DataFrame(result)
        result = result.rename(columns={"gender": "gender_infered"})
        if len(result) == len(self.test_data):
            self.test_data = pd.concat([self.test_data, result], axis=1)
        else:
            print("response from genderize.io contains less results than request. Try again?")
        self.test_data.drop("name", axis=1, inplace=True)
        self.test_data.replace(to_replace={"gender_infered": {'male': 'm', "female": "f", None: "u"}}, inplace=True)
        self.gender_evaluator = 'genderize_io'

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
