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

        try:
            for row in self.test_data.itertuples():
                if row.middle_name == '':
                    self.api_response.extend(Genderize().get([row.first_name]))
                    print(self.api_response[-1])
                else:  # if middle_name exists then try various variations of full name
                    connectors = ['', ' ', '-']
                    names = [row.first_name + c + row.middle_name for c in connectors]
                    print("middlenames exist")
                    print(names)
                    api_resp = Genderize().get(names)
                    if set([r['gender'] for r in api_resp]) == {None}:
                        print("middlename no help")
                        self.api_response.extend(Genderize().get([row.first_name]))
                        print(self.api_response[-1])
                    else:  # if usage of middle name leads to female or male then take assignment with highest count
                        for item in api_resp:
                            if item['gender'] is None:
                                item['count'], item['probability'] = 0, 0.0
                        names_to_responses = dict(zip(names, api_resp))
                        print("middlename helped")
                        print(names_to_responses)
                        names_to_responses = OrderedDict(
                            sorted(names_to_responses.items(), key=lambda x: x[1]['count'], reverse=True))
                        self.api_response.append(
                            next(iter(names_to_responses.values())))  # select first item in ordered dict

            self.extend_test_data_by_api_response(self.api_response, {'male': 'm', "female": "f", None: "u"})

        except GenderizeException as e:
            print(e)
