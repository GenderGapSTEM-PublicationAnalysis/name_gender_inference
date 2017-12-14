from evaluator import Evaluator
from genderize import Genderize, GenderizeException
from collections import OrderedDict


# TODO: get API key
class GenderizeIoEvaluator(Evaluator):
    gender_evaluator = 'genderize_io'

    def __init__(self, file_path):
        Evaluator.__init__(self, file_path)

    def _fetch_gender_from_api(self):
        """Fetches gender predictions from genderize.io using self.test_data.first_name and
        self.test_data.middle_name. Results are stored in self.api_response as a list.
        If result list complete then they are merged with self.test_data."""

        for row in self.test_data.itertuples():
            try:
                if row.middle_name == '':
                    self.api_response.extend(Genderize().get([row.first_name]))
                else:  # if middle_name exists then try various variations of full name
                    connectors = ['', ' ', '-']
                    names = [row.first_name + c + row.middle_name for c in connectors]
                    api_resp = Genderize().get(names)
                    if set([r['gender'] for r in api_resp]) == {None}:
                        self.api_response.extend(Genderize().get([row.first_name]))
                    else:  # if usage of middle name leads to female or male then take assignment with highest count
                        for item in api_resp:
                            if item['gender'] is None:
                                item['count'], item['probability'] = 0, 0.0
                        names_to_responses = dict(zip(names, api_resp))
                        names_to_responses = OrderedDict(
                            sorted(names_to_responses.items(), key=lambda x: x[1]['count'], reverse=True))
                        self.api_response.append(
                            next(iter(names_to_responses.values())))  # select first item in ordered dict

            except GenderizeException as e:
                print(e)

            self.extend_test_data_by_api_response(self.api_response, {'male': 'm', "female": "f", None: "u"})
