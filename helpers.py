import sys
import pandas as pd


def clean_name_part(df, name_part="middle_name"):
    """keep the string in column 'middle_name' if it has more than one character.
    Otherwise replace by ''. """

    def try_to_simplify(s):
        try:
            if len(s) > 1:
                return s.lower()
            else:
                return ''
        except:
            return ''

    df[name_part] = df[name_part].map(lambda x: try_to_simplify(x))


def build_full_name(df):
    df["full_name"] = df.apply(lambda x: x.first_name + ' ' + x.middle_name + ' ' + x.last_name, axis=1)
    df.full_name = df.full_name.str.replace('  ', ' ')  # if no middle_name then the above line yields 2 empty spaces


class memoize:
    # from http://avinashv.net/2008/04/python-decorators-syntactic-sugar/
    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]


# Taken from http://scottlobdell.me/2015/08/using-decorators-python-automatic-registration/
REGISTERED_EVALUATORS = []


def register_evaluator(cls):
    REGISTERED_EVALUATORS.append(cls)
    return cls


def show_progress(row_index):
    """Shows a progress bar"""
    if row_index % 100 == 0:
        sys.stdout.write('{}...'.format(row_index))
        sys.stdout.flush()


# TODO: remove in the end, probably not required
def compute_equal_frequency_binning(param_values, k):
    """Takes a list of values (e.g. of a tuning parameter) and an integer k, and returns the lower
    quantile boundaries for k quantiles. Corresponds to equal-frequency binning. Take 0 always into results."""
    bins = pd.qcut(param_values, k, duplicates='drop')
    left_boundaries = [0]

    for b in bins:
        left_boundaries.append(b.left)
    return sorted(list(set(left_boundaries)))
