import sys
import json
from collections import OrderedDict

import gender_guesser.detector as gender
import requests
from urllib.request import urlopen

from urllib.parse import urlencode
from genderize import Genderize, GenderizeException

from evaluator import Evaluator
from helpers import memoize


def show_progress(row_index):
    """Shows a progress bar"""
    if row_index % 100 == 0:
        sys.stdout.write('{}...'.format(row_index))
        sys.stdout.flush()


class GenderAPIEvaluator(Evaluator):
    gender_evaluator = 'gender_api'
    api_key = 'TMbKcgUmgSpBtnjWoT'  # TODO: obfuscate key if we make package open

    def __init__(self, data_source):
        Evaluator.__init__(self, data_source)

    @staticmethod
    @memoize
    def call_api(n, verb='split'):
        urlpars = urlencode({'key': GenderAPIEvaluator.api_key, verb: n})
        url = 'https://gender-api.com/get?{}'.format(urlpars)
        response = urlopen(url)
        decoded = response.read().decode('utf-8')
        return json.loads(decoded)

    def _fetch_gender_from_api(self):
        # if api_response already contains partial results then do not re-evaluate them
        start_position = len(self.api_response)
        names = self.test_data[start_position:].full_name.tolist()
        for i, n in enumerate(names):
            show_progress(i)

            # This implementation is for full_name
            data = GenderAPIEvaluator.call_api(n)
            if 'errmsg' not in data.keys():
                self.api_response.append(data)
            else:
                break
        self.extend_test_data_by_api_response(self.api_response,
                                              {'male': 'm', 'female': 'f', 'unknown': 'u'})


class GenderAPIPieceEvaluator(GenderAPIEvaluator):
    gender_evaluator = 'gender_api_piece'

    def __init__(self, data_source):
        GenderAPIEvaluator.__init__(self, data_source)

    def _fetch_gender_from_api(self):

        # if api_response already contains partial results then do not re-evaluate them
        start_position = len(self.api_response)

        for i, row in enumerate(self.test_data[start_position:].itertuples()):
            # Print sort of progress bar
            show_progress(i)

            # This implementation is for name pieces
            if row.middle_name == '' or row.first_name == '':
                # If one of the forenames is missing, try just the other
                name = row.middle_name or row.first_name
                data = GenderAPIPieceEvaluator.call_api(name, verb='name')
            else:
                # If middle name, try various combinations
                connectors = ['', ' ', '-']
                names = [c.join([row.first_name, row.middle_name]) for c in connectors]
                api_resp = [GenderAPIPieceEvaluator.call_api(n, verb='name') for n in names]
                if set([r['gender'] for r in api_resp]) == {None}:
                    # If no gender with both names, try first only
                    data = GenderAPIPieceEvaluator.call_api(row.first_name, verb='name')
                    self.api_response.append(data)
                else:
                    # if usage of middle name leads to female or male then take assignment with highest samples
                    data = max(api_resp, key=lambda x: x['samples'])
                    self.api_response.append(data)
            if 'errmsg' not in data.keys():
                self.api_response.append(data)
            else:
                break
        self.extend_test_data_by_api_response(self.api_response,
                                              {'male': 'm', 'female': 'f', 'unknown': 'u'})


# Used this blog post: https://juliensalinas.com/en/REST_API_fetching_go_golang_vs_python/
# linked from the API's website: https://www.nameapi.org/en/developer/downloads/
class NamesAPIEvaluator(Evaluator):
    api_key = "725a6a1ddf0d0f16f7dc3a6a73a9ac5b-user1"
    gender_evaluator = 'names_api'
    url = "http://rc50-api.nameapi.org/rest/v5.0/genderizer/persongenderizer?apiKey="

    def __init__(self, data_source):
        Evaluator.__init__(self, data_source)

    @staticmethod
    @memoize
    def call_api(name):
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

        query = build_json(name)
        url = NamesAPIEvaluator.url + NamesAPIEvaluator.api_key
        resp = requests.post(url, json=query)
        resp.raise_for_status()
        return resp.json()

    def _fetch_gender_from_api(self):

        start_position = len(
            self.api_response)  # if api_response already contains partial results then do not re-evaluate them
        names = self.test_data[start_position:].full_name.tolist()
        for i, n in enumerate(names):
            show_progress(i)
            try:
                self.api_response.append(NamesAPIEvaluator.call_api(n))
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

    @staticmethod
    @memoize
    def call_api(n):
        return gender.Detector().get_gender(n)

    def _fetch_gender_from_api(self):
        # exact response stored in column `response`. This can be tuned using training data
        start_position = len(self.api_response)
        for i, row in enumerate(self.test_data[start_position:].itertuples()):
            show_progress(i)
            if row.middle_name != '':
                name = row.first_name.title() + '-' + row.middle_name.title()
                g = GenderGuesserEvaluator.call_api(name)
                if g != "unknown":
                    self.api_response.append(g)
                else:
                    name = row.first_name.title()
                    self.api_response.append(GenderGuesserEvaluator.call_api(name))
            else:
                name = row.first_name.title()
                self.api_response.append(GenderGuesserEvaluator.call_api(name))

        self.test_data["gender_infered"] = self.api_response
        self.test_data["response"] = self.api_response
        self.test_data.replace(to_replace={"gender_infered": {'male': 'm', "female": "f", "mostly_male": "m",
                                                              "mostly_female": "f", "unknown": "u", "andy": "u"}},
                               inplace=True)


class GenderizeIoEvaluator(Evaluator):
    gender_evaluator = 'genderize_io'

    def __init__(self, data_source):
        Evaluator.__init__(self, data_source)

    @staticmethod
    @memoize
    def call_api(names):
        return Genderize().get(names)

    def _fetch_gender_from_api(self):
        """Fetches gender predictions from genderize.io using self.test_data.first_name and
        self.test_data.middle_name. Results are stored in self.api_response as a list.
        If result list complete then they are merged with self.test_data."""

        start_position = len(self.api_response)
        for i, row in enumerate(self.test_data[start_position:].itertuples()):
            # Print sort of progress bar
            show_progress(i)
            try:
                if row.middle_name == '':
                    self.api_response.extend(GenderizeIoEvaluator.call_api([row.first_name]))
                else:  # if middle_name exists then try various variations of full name
                    connectors = ['', ' ', '-']
                    names = [row.first_name + c + row.middle_name for c in connectors]
                    api_resp = GenderizeIoEvaluator.call_api(names)
                    if set([r['gender'] for r in api_resp]) == {None}:
                        self.api_response.extend(GenderizeIoEvaluator.call_api([row.first_name]))
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
