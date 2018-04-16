# coding=utf-8
from setuptools import setup, find_packages

reqired_pckgs = []

setup(name='name_gender_inference',
      version='0.1',
      description='Code and data for evaluation of name-based gender inference tools',
      url='https://github.com/GenderGapSTEM-PublicationAnalysis/name-gender-inference',
      author='Helena Mihaljević and Lucia Santamaría',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      python_requires='>=3.5',
      install_requires=reqired_pckgs,
      zip_safe=False)
