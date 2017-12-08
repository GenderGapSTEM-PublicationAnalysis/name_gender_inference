from evaluator import Evaluator
import gender_guesser.detector as gender
import pandas as pd


class GenderGuesserEvaluator(Evaluator):
    gender_evaluator = 'gender_guesser'  # based on Joerg Michael's C-program `gender`

    def __init__(self, file_path):
        Evaluator.__init__(self, file_path)

    def _fetch_gender_from_api(self):
        # exact response stored in column `response`. This can be tuned using training data
        # TODO: implement training
        names = self.test_data.first_name.tolist()
        result = []
        for n in names:
            result.append(gender.Detector().get_gender(n.title()))

        self.test_data["gender_infered"] = result
        self.test_data["response"] = result
        self.test_data.replace(to_replace={"gender_infered": {'male': 'm', "female": "f", "mostly_male": "m",
                                                              "mostly_female": "f", "unknown": "u", "andy": "u"}},
                               inplace=True)
