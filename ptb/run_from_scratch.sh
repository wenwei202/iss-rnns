#!/bin/bash
set -e
set -x

DATASET_NAME=ptb
ROOT_WORKSPACE=${HOME}/trained_models/${DATASET_NAME}/ # the location to store summary and logs
DATA_DIR=${HOME}/github/users/wenwei202/models/tutorials/rnn/ptb/simple-examples/data/ # dataset location
export CUDA_VISIBLE_DEVICES=0 # specify visible gpus to tensorflow
NET=sparselarge
OPTIMIZER=gd
BASE_LR=1.0
WEIGHT_DECAY=0.00011
DROPOUT_KEEP=0.65

if [ ! -d "$ROOT_WORKSPACE" ]; then
  echo "${ROOT_WORKSPACE} does not exsit!"
  exit
fi

TRAIN_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_training_data/
INFO_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_log/
if [ ! -d "${INFO_WORKSPACE}" ]; then
  echo "Creating ${INFO_WORKSPACE} ..."
  mkdir -p ${INFO_WORKSPACE}
fi
current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}
FOLDER_NAME=${DATASET_NAME}_${NET}_${OPTIMIZER}_${BASE_LR}_${WEIGHT_DECAY}_${DROPOUT_KEEP}_${current_time}
TRAIN_DIR=${TRAIN_WORKSPACE}/${FOLDER_NAME}
if [ ! -d "$TRAIN_DIR" ]; then
  echo "Creating ${TRAIN_DIR} ..."
  mkdir -p ${TRAIN_DIR}
fi

python ptb_word_lm.py \
--model ${NET} \
--optimizer ${OPTIMIZER} \
--weight_decay ${WEIGHT_DECAY} \
--dropout_keep ${DROPOUT_KEEP} \
--data_path ${DATA_DIR} \
--save_path ${TRAIN_DIR}  >  ${INFO_WORKSPACE}/${FOLDER_NAME}.log 2>&1 &
