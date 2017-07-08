#!/bin/bash
source activate py27

hyperparameters=$1
log_path=$2

python -u ptb_word_lm_hd.py --model large --data_path simple-examples/data/ --save_path ${log_path}  --config_file ${hyperparameters} &> ${log_path}/training.log
