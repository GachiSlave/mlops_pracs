#!/bin/bash
pip install -r lab1/requirements.txt
python3 lab1/data_creation.py
python3 lab1/model_preparation.py
python3 lab1/model_preprocessing.py
python3 lab1/model_testing.py
