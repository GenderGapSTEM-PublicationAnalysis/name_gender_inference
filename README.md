# README #


## What is this repository for? ##

This repository provides data and code for evaluating various methods that infer a
person's gender from the name string.
Currently, evaluator classes are implemented for four web services, [Gender API](https://gender-api.com/),
[Genderize.Io](https://genderize.io/), [NameAPI](https://www.nameapi.org) and [NamSor](https://api.namsor.com),
and for the Python package *gender-guesser*.

## Evaluator classes


## How to fetch gender from an API

Use the two variables `data_source` and `service_name` below to select a data set and an evaluator object, respectively, e.g.:

```python
from evaluators import *
data_source = 'nature'
service_name = GenderizeIoEvaluator

```

Initialize an evaluator object and load the data from the file coded by
## Extend code by new Evaluator class

Need to create a sub-directory of `test_data/raw_data` having the same name as the evaluator (class attribute `gender_evalauator`)
with write permission