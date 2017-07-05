#!/bin/bash
export OUTPUT_DIR=$1

python -m basic.cli --len_opt --cluster \
--shared_path ${OUTPUT_DIR}/basic/00/shared.json \
--load_path ${OUTPUT_DIR}/basic/00/save/basic-10000 \
--num_gpus 2 --batch_size 30 \
--plot_weights \
--group_config groups.json

python squad/evaluate-v1.1.py \
$HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-000000.json
