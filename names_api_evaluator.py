from evaluator import Evaluator
import pandas as pd
import requests


class NamesAPIEvaluator(Evaluator):
    api_key = "725a6a1ddf0d0f16f7dc3a6a73a9ac5b-user1"
    gender_evaluator = 'names_api'

    def __init__(self, file_path):
        Evaluator.__init__(self, file_path)

    def _fetch_gender_from_api(self):
        names = self.test_data.full_name.tolist()

        def build_json(name):
            return {
                "inputPerson": {
                    "type": "NaturalInputPerson",
                    "personName": {
                        "nameFields": [
                            {
                                "string": name,
                                "fieldType": "FULLNAME"
                            }
                        ]
                    }
                }
            }

        def build_url(key=self.api_key):
            return "http://rc50-api.nameapi.org/rest/v5.0/genderizer/persongenderizer?apiKey=" + key

        def build_dataframe_from_response(data, result):
            result = pd.DataFrame(result).rename(columns={"gender": "gender_infered"})
            data = pd.concat([data, result], axis=1)
            return data.replace(to_replace={"gender_infered": {'MALE': 'm', 'FEMALE': 'f', 'UNKNOWN': 'u'}})

        responses = []
        error_response = {'gender': 'error', 'confidence': 1.0}
        url = build_url()
        for n in names:
            try:
                query = build_json(n)
                resp = requests.post(url, json=query)
                resp.raise_for_status()
                # Decode JSON response into a Python dict:
                resp_dict = resp.json()
                print(resp_dict)
                responses.append(resp_dict)
            except requests.exceptions.HTTPError as e:
                print("Bad HTTP status code:", e)
                responses.append(error_response)
            except requests.exceptions.RequestException as e:
                print("Network error:", e)
                responses.append(error_response)

        self.test_data = build_dataframe_from_response(self.test_data, responses)
