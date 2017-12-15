from evaluator import Evaluator
import gender_guesser.detector as gender


class GenderGuesserEvaluator(Evaluator):
    gender_evaluator = 'gender_guesser'  # based on Joerg Michael's C-program `gender`

    def __init__(self, data_source):
        Evaluator.__init__(self, data_source)

    def _fetch_gender_from_api(self):
        # exact response stored in column `response`. This can be tuned using training data
        start_position = len(self.api_response)
        for row in self.test_data[start_position:].itertuples():
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
