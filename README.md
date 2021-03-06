# Evaluation of name-based gender inference methods #

# README #

## What is this repository for? ##

This repository provides data and code for evaluating various methods that infer a
person's gender from the name string.
Currently, evaluator classes are implemented for four web services, [Gender API](https://gender-api.com/),
[Genderize.Io](https://genderize.io/), [NameAPI](https://www.nameapi.org) and [NamSor](https://api.namsor.com),
and for the Python package *gender-guesser*.

## Evaluator classes and data sources

In the file `evaluators.py`, a class is defined for each gender inference service.
All classes inherit from the ABC-class `Evaluator` defined in `evaluator.py` (which cannot be instantiated).

On an evaluator object, the following actions can be performed:

1. load a data file with names and gender
2. retrieve the gender for the names by querying the service
3. store names with gender assignments from service
4. compute error metrics for the inferred gender
5. tune free parameters such as accuracy provided by the service and compute train-/test-errors


To initialize an evaluator object, you need to provide the 'name' of a data source.
The functions for loading and storing data use a naming convention for data source files; see `Evaluator.__init__`.

For instance, to query the service <genderize.io> with all names from the data set 'genderize_r_authors',
instantiate an object and load the file as follows:

```python
from evaluators import GenderizeIoEvaluator

evaluator = GenderizeIoEvaluator('genderize_r_authors')
evaluator.load_data()
```
This will load the file `test_data/raw_data/genderize_r_authors.csv` that contains the corresponding raw
 data set as a `pandas DataFrame` into the attribute `evaluator.test_data`.
 The method `Evaluator.dump_evaluated_test_data_to_file` can then be used to store evaluations of
 this data source as `test_data/genderize_io/genderize_r_authors_genderize_io.csv`. Once a data source has been
 evaluated, one can load the file with evaluations by providing the option `evaluated=True` to the load function.

The naming convention expects all data files to be in the directory `test_data`.
Therein, the folder `raw_data` contains files with test data that can be used to evaluate a service,
and each other directory contains the evaluated files per service.

Currently, the following data sources are available:

* `genderize_r_authors`: Authorships dataset downloaded from [here](https://github.com/kalimu/genderizeR/tree/master/data); see the article
by [K. Wais (2016)](https://journal.r-project.org/archive/2016/RJ-2016-002/index.html) for a description of the data set.
* `genderize_r_titles`: Dataset with titles of biographical articles downloaded from [here](https://github.com/kalimu/genderizeR/tree/master/data); see the article
by [K. Wais (2016)](https://journal.r-project.org/archive/2016/RJ-2016-002/index.html) for a description of the data set.
* `wos`: Data set produced for the [validation study](https://www.nature.com/polopoly_fs/7.14227.1386700530!/suppinfoFile/504211a_s1.pdf)
of the findings of [V. Lariviere et al. (2013)](http://dx.doi.org/10.1038/504211a),
consisting of 1,000 randomly chosen records from the WoS database covering publications from 2008 to 2012
included in Science Citation Index Expanded, the Social Sciences Citation Index and the Arts and Humanities Citation Index.
* `pubmed`: Authorships from articles published between 02/1994 and 06/2014 in the 6 journals with highest
JCR impact factor in 2012 in the category ``Medicine, general \& internal''.
See the article by [G. Filardo et al. (2-16)](https://10.1136/bmj.i847) for a description of the data set.
* `zbmath`: 400 randomly selected authors from the bibliographical records of the mathematical publications service zbMATH
 that was used in the article by [H. Mihaljević-Brandt et al. (2016)](https://doi.org/10.1371/journal.pone.0165367)
 to evaluate the gender inference method.
* `all`: Concatenation of all other data sources

## Fetch gender from a service

Use the method `Evaluator.fetch_gender` to query the chosen service:


```python
from evaluators import GenderizeIoEvaluator

evaluator = GenderizeIoEvaluator('genderize_r_authors')
evaluator.load_data()
evaluator.fetch_gender()

```

The service responses are then stored in `evaluator.api_response`, e.g.

```python
[
    {'count': 7, 'gender': 'male', 'name': 'onno', 'probability': 1.0},
    {'gender': None, 'name': 'badugu'},
    {'count': 3, 'gender': 'male', 'name': 'abolfazl', 'probability': 1.0},
    {'count': 1, 'gender': 'male', 'name': 'ejiofor', 'probability': 0.95}
]
```

If the request to the service is successful then the gender responses are by default merged into the `evaluator.test_data`
attribute and the dump-method is called. Otherwise, call this method with the option `save_to_dump=False`.

If the query to the service could not be completed for all names, the partial response is still stored in
`evaluator.api_response`; calling `fetch_gender()` again will attempt to query the service for
names that have not be successfully queried.

