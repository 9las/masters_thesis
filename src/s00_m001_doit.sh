#!/bin/bash

./s01_e01_train_job_parallel.sh
./s02_predict.py
./s03_calculate_model_performance.py
./s04_plot_model_performance.R
