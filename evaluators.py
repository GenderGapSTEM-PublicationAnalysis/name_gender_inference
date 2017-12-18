# TODO: check error messages when catching exceptions before publishing code
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
    """This implementation is for using name pieces"""
    gender_evaluator = 'gender_api'
    api_key = 'HjmUptFvSCCbSlHPkP'  # TODO: obfuscate key if we make package open

    def __init__(self, data_source):
        Evaluator.__init__(self, data_source)

    @staticmethod
    @memoize
    def _call_api(n):
        urlpars = urlencode({'key': GenderAPIEvaluator.api_key, 'name': n})
        url = 'https://gender-api.com/get?{}'.format(urlpars)
        response = urlopen(url)
        decoded = response.read().decode('utf-8')
        return json.loads(decoded)

    def _fetch_gender_from_api(self):
        start_position = len(self.api_response)

        for i, row in enumerate(self.test_data[start_position:].itertuples()):
            show_progress(i)
            try:
                if row.middle_name == '':
                    # If middle name is missing, try just first_name alone
                    data = GenderAPIEvaluator._call_api(row.first_name)
                else:
                    # If middle name, try various combinations
                    connectors = ['', ' ', '-']
                    names = [c.join([row.first_name, row.middle_name]) for c in connectors]
                    api_resp = [GenderAPIEvaluator._call_api(n) for n in names]
                    if set([r['gender'] for r in api_resp]) == {'unknown'}:
                        # If no gender with both names, try first only
                        data = GenderAPIEvaluator._call_api(row.first_name)
                        # self.api_response.extend(data)
                    else:
                        # if usage of middle name leads to female or male then take assignment with highest samples
                        data = max(api_resp, key=lambda x: x['samples'])
                        # self.api_response.append(data)
                if 'errmsg' not in data.keys():
                    self.api_response.append(data)
                else:
                    break
            except:
                print("Some unexpected error occured")
        self.extend_test_data_by_api_response(self.api_response,
                                              {'male': 'm', 'female': 'f', 'unknown': 'u'})


class GenderAPIFullEvaluator(GenderAPIEvaluator):
    """This implementation is for full_name"""
    gender_evaluator = 'gender_api_full'

    def __init__(self, data_source):
        GenderAPIEvaluator.__init__(self, data_source)

    @staticmethod
    @memoize
    def _call_api(n):
        urlpars = urlencode({'key': GenderAPIEvaluator.api_key, 'split': n})
        url = 'https://gender-api.com/get?{}'.format(urlpars)
        response = urlopen(url)
        decoded = response.read().decode('utf-8')
        return json.loads(decoded)

    def _fetch_gender_from_api(self):
        start_position = len(self.api_response)
        names = self.test_data[start_position:].full_name.tolist()
        for i, n in enumerate(names):
            show_progress(i)
            try:
                data = GenderAPIEvaluator._call_api(n)
                if 'errmsg' not in data.keys():
                    self.api_response.append(data)
                else:
                    break
            except:
                print("An unexpected error occured")
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
    def _call_api(name):
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
                self.api_response.append(NamesAPIEvaluator._call_api(n))
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
    def _call_api(n):
        return gender.Detector().get_gender(n)

    def _fetch_gender_from_api(self):
        # exact response stored in column `response`. This can be tuned using training data
        start_position = len(self.api_response)
        for i, row in enumerate(self.test_data[start_position:].itertuples()):
            show_progress(i)
            if row.middle_name != '':
                name = row.first_name.title() + '-' + row.middle_name.title()
                g = GenderGuesserEvaluator._call_api(name)
                if g != "unknown":
                    self.api_response.append(g)
                else:
                    name = row.first_name.title()
                    self.api_response.append(GenderGuesserEvaluator._call_api(name))
            else:
                name = row.first_name.title()
                self.api_response.append(GenderGuesserEvaluator._call_api(name))

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
    def _call_api(names):
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
                    self.api_response.extend(GenderizeIoEvaluator._call_api([row.first_name]))
                else:  # if middle_name exists then try various variations of full name
                    connectors = ['', ' ', '-']
                    names = [row.first_name + c + row.middle_name for c in connectors]
                    api_resp = GenderizeIoEvaluator._call_api(names)
                    if set([r['gender'] for r in api_resp]) == {None}:
                        self.api_response.extend(GenderizeIoEvaluator._call_api([row.first_name]))
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
