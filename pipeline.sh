#!/bin/bash

# Запуск Python-скриптов по порядку
python data_creation.py
python model_preprocessing.py
python model_preparation.py
python model_testing.py
