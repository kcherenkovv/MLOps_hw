#!/bin/bash

# Запуск Python-скриптов по порядку
python3 scripts/data_creation.py
python3 scripts/model_preprocessing.py
python3 scripts/model_preparation.py
python3 scripts/model_testing.py
