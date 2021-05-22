# Synthetic Electrocardiogram Attack Method

Synthetic Electrocardiogram Attack Method (SEAM) is an attack designed to compromise electrocardiogram based key agreement schemes for Body Area Networks (BAN) such as ELPA, PSKA or OPFKA. This attack will focus on the property known as temporal variance which prevents the use of past signal recordings to be used against present or future agreements. The attack will attempt to reconstitute the QRS complexes that originate from past recordings in possession of the adversary in order to achieve this aim.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

Download a compatible dataset such as the [MIT-BIH Normal Sinus Rhythm Dataset](https://physionet.org/content/nsrdb/1.0.0/) and extract to the datasets folder in the root of this project

Run pip install -r requirements.txt to install packages

Construct a new database use the provided schema database.sql and then run main.py

or 

Install jupyter notebook on top of the requirements and run Synthetic Electrocardiogram Attack Method.ipynb

## Dependencies

For the most up-to-date list of dependencies see requirements.txt 

## License

The MIT License (MIT)

Copyright (C) 2021 Jack Hodgkiss

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.