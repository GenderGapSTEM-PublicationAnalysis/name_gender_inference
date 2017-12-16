import sys
from collections import OrderedDict

import gender_guesser.detector as gender
import requests
from genderize import Genderize, GenderizeException

from evaluator import Evaluator


# Used this blog post: https://juliensalinas.com/en/REST_API_fetching_go_golang_vs_python/
# linked from the API's website: https://www.nameapi.org/en/developer/downloads/

class NamesAPIEvaluator(Evaluator):
    api_key = "725a6a1ddf0d0f16f7dc3a6a73a9ac5b-user1"
    gender_evaluator = 'names_api'

    def __init__(self, data_source):
        Evaluator.__init__(self, data_source)

    def _fetch_gender_from_api(self):

        def build_json(name):
            return {
                "inputPerson": {
                    "type": "NaturalInputPerson",
                    "personName": {
                        "nameFields": [
                            {
                                "string": name.title(),
                                "fieldType": "FULLNAME"
                            }
                        ]
                    }
                }
            }

        def build_url(key=self.api_key):
            return "http://rc50-api.nameapi.org/rest/v5.0/genderizer/persongenderizer?apiKey=" + key

        url = build_url()
        start_position = len(
            self.api_response)  # if api_response already contains partial results then do not re-evaluate them
        names = self.test_data[start_position:].full_name.tolist()
        for i, n in evaluate(names):
            # Print sort of progress bar
            if i % 100 == 0:
                sys.stdout.write('{}...'.format(i))
                sys.stdout.flush()
            try:
                query = build_json(n)
                resp = requests.post(url, json=query)
                resp.raise_for_status()
                # Decode JSON response into a Python dict:
                resp_dict = resp.json()
                self.api_response.append(resp_dict)
            except requests.exceptions.HTTPError as e:
                print("Bad HTTP status code:", e)
                break
            except requests.exceptions.RequestException as e:
                print("Network error:", e)
                break
        self.extend_test_data_by_api_response(self.api_response,
                                              {'MALE': 'm', 'FEMALE': 'f', 'UNKNOWN': 'u', 'NEUTRAL': 'u'})


class GenderGuesserEvaluator(Evaluator):
    gender_evaluator = 'gender_guesser'  # based on Joerg Michael's C-program `gender`

    def __init__(self, data_source):
        Evaluator.__init__(self, data_source)

    def _fetch_gender_from_api(self):
        # exact response stored in column `response`. This can be tuned using training data
        start_position = len(self.api_response)
        for i, row in enumerate(self.test_data[start_position:].itertuples()):
            # Print sort of progress bar
            if i % 100 == 0:
                sys.stdout.write('{}...'.format(i))
                sys.stdout.flush()
            if row.middle_name != '':
                name = row.first_name.title() + '-' + row.middle_name.title()
                g = gender.Detector().get_gender(name)
                if g != "unknown":
                    self.api_response.append(g)
                else:
                    name = row.first_name.title()
                    self.api_response.append(gender.Detector().get_gender(name))
            else:
                name = row.first_name.title()
                self.api_response.append(gender.Detector().get_gender(name))

        self.test_data["gender_infered"] = self.api_response
        self.test_data["response"] = self.api_response
        self.test_data.replace(to_replace={"gender_infered": {'male': 'm', "female": "f", "mostly_male": "m",
                                                              "mostly_female": "f", "unknown": "u", "andy": "u"}},
                               inplace=True)


# TODO: get API key
class GenderizeIoEvaluator(Evaluator):
    gender_evaluator = 'genderize_io'

    def __init__(self, data_source):
        Evaluator.__init__(self, data_source)

    def _fetch_gender_from_api(self):
        """Fetches gender predictions from genderize.io using self.test_data.first_name and
        self.test_data.middle_name. Results are stored in self.api_response as a list.
        If result list complete then they are merged with self.test_data."""

        start_position = len(self.api_response)
        for i, row in enumerate(self.test_data[start_position:].itertuples()):
            # Print sort of progress bar
            if i % 100 == 0:
                sys.stdout.write('{}...'.format(i))
                sys.stdout.flush()
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
                break

        self.extend_test_data_by_api_response(self.api_response, {'male': 'm', "female": "f", None: "u"})
