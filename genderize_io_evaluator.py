from evaluator import Evaluator
from genderize import Genderize, GenderizeException
import pandas as pd


class GenderizeIoEvaluator(Evaluator):
    gender_evaluator = 'genderize_io'

    def __init__(self, file_path):
        Evaluator.__init__(self, file_path)

    def _fetch_gender_from_api(self):
        """Fetches gender predictions from genderize.io using self.test_data.first_name
                and merges them with self.test_data."""
        print("Calling the right method")  # TODO: remove after testing the notebook
        names = self.test_data.first_name.tolist()
        result = []
        i = 0
        try:
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
            self.test_data.replace(to_replace={"gender_infered": {'male': 'm', "female": "f",
                                                                  None: "u"}}, inplace=True)
            self.gender_evaluator = 'genderize_io'
        except GenderizeException as e:
            print(e)
