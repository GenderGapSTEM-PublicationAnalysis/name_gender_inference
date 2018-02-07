from glob import glob
from os.path import basename, dirname, join, abspath

from helpers import REGISTERED_EVALUATORS

DIR_PATH = dirname(abspath(__file__))

# Taken from http://scottlobdell.me/2015/08/using-decorators-python-automatic-registration/
# Import all classes in this directory so that classes with @register_evaluator are registered.
pwd = dirname(__file__)
for x in glob(join(pwd, '*.py')):
    if not x.startswith('__'):
        __import__(basename(x)[:-3], globals(), locals())

__all__ = [
    'REGISTERED_EVALUATORS'
]
