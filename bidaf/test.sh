#!/bin/bash
set -e
set -x

if [ "$#" -lt 3 ]; then
	echo "Illegal number of parameters"
	echo "Usage: $0 OUTPUT_DIR HIDDEN_SIZE GROUP_CONFIG"
	exit
fi


OUTPUT_DIR=$1
HIDDEN_SIZE=$2
GROUP_CONFIG=$3

python -m basic.cli --len_opt --cluster \
--shared_path ${OUTPUT_DIR}/basic/00/shared.json \
--load_path ${OUTPUT_DIR}/basic/00/save/basic-10000 \
--num_gpus 2 --batch_size 30 \
--hidden_size ${HIDDEN_SIZE} \
--plot_weights \
--zero_threshold 0.0004 \
--group_config ${GROUP_CONFIG}

python squad/evaluate-v1.1.py \
$HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-000000.json
