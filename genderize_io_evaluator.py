from evaluator import Evaluator
from genderize import Genderize, GenderizeException
from collections import OrderedDict


# TODO: get API key
class GenderizeIoEvaluator(Evaluator):
    gender_evaluator = 'genderize_io'

    def __init__(self, file_path):
        Evaluator.__init__(self, file_path)

    def _fetch_gender_from_api(self):
        """Fetches gender predictions from genderize.io using self.test_data.first_name
                and merges them with self.test_data."""

        responses = []
        try:
            for row in self.test_data.itertuples():
                if row.middle_name == '':
                    responses.extend(Genderize().get([row.first_name]))
                else:  # if middle_name exists then try various variations of full name
                    connectors = ['', ' ', '-']
                    names = [row.first_name + c + row.middle_name for c in connectors]
                    api_resp = Genderize().get(names)
                    if set([r['gender'] for r in api_resp]) == {None}:
                        responses.extend(Genderize().get([row.first_name]))
                    else:  # if usage of middle name leads to female or male then take assignment with highest count
                        for item in api_resp:
                            if item['gender'] is None:
                                item['count'], item['probability'] = 0, 0.0
                        names_to_responses = dict(zip(names, api_resp))
                        names_to_responses = OrderedDict(
                            sorted(names_to_responses.items(), key=lambda x: x[1]['count'], reverse=True))
                        responses.append(next(iter(names_to_responses.values())))  # select first item in ordered dict

            self.api_response = responses
            self.extend_test_data_by_api_response(responses, {'male': 'm', "female": "f", None: "u"})

        except GenderizeException as e:
            print(e)
