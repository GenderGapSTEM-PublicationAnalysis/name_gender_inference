from evaluator import Evaluator
import pandas as pd


class NamesAPIEvaluator(Evaluator):
    api_key = "bbbd8f9ba16f58f69ef21f8b6509aac8-user1"
    gender_evaluator = 'names_api'

    def __init__(self, file_path):
        Evaluator.__init__(self, file_path)



        # def _fetch_gender_from_api(self):
        #     names = self.test_data.
        #     def build_json(name):
        #         return {
        #             "inputPerson": {
        #                 "type": "NaturalInputPerson",
        #                 "personName": {
        #                     "nameFields": [
        #                         {
        #                             "string": name,
        #                             "fieldType": "FULLNAME"
        #                         }
        #                     ]
        #                 }
        #             }
        #         }
        #
        #     def build_url(api_key=NAMEAPI_KEY):
        #         return "http://rc50-api.nameapi.org/rest/v5.0/genderizer/persongenderizer?apiKey=" + api_key
        #
        #     responses = []
        #     error_response = {'gender': 'error', 'confidence': 1.0}
        #     url = build_url()
        #     for n in names:
        #         try:
        #             query = build_json(n)
        #             resp = requests.post(url, json=query)
        #             resp.raise_for_status()
        #             # Decode JSON response into a Python dict:
        #             resp_dict = resp.json()
        #             print(resp_dict)
        #             responses.append(resp_dict)
        #         except requests.exceptions.HTTPError as e:
        #             print("Bad HTTP status code:", e)
        #             responses.append(error_response)
        #         except requests.exceptions.RequestException as e:
        #             print("Network error:", e)
        #             responses.append(error_response)
