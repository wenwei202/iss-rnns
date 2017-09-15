# Learning Intrinsic Sparse Structures in BiDAF
 
- This is a modified implementation of [Bi-directional Attention Flow for Machine Comprehension][paper] (Seo et al., 2016).
- This is tensorflow v1.1.0 comaptible version. 

Use `python -m basic.cli --help` for usage.

## Requirements
#### General
- Python (developed on 3.5.2. Issues have been reported with Python 2!)
- unzip

#### Python Packages
- tensorflow (deep learning library, verified on 1.1.0)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visaulization; if you only train and test, not needed)

## Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```

Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
```
python -m squad.prepro
```

## Training BiDAF baseline
Note that the training script saves results in a subfolder of `out` named as `${TIMESTAMP}` (e.g. `out/2017-07-10___21-37-44/`). Before running, please:
```
mkdir out
```
The model requires at least 12GB of GPU RAM.
If your GPU RAM is smaller than 12GB, you can either decrease batch size (performance might degrade),
or you can use multi GPU (see below).
The training converges at ~10k steps, and it took ~10 hours in Titan X.

Before training, it is recommended to first try the following code to verify everything is okay and memory is sufficient:
```
python -m basic.cli --mode train --noload --debug
```

Then to fully train baseline without sparsity learning, run:
```
python -m basic.cli --mode train --noload
```

You can speed up the training process with optimization flags:
```
python -m basic.cli --mode train --noload --len_opt --cluster
```
You can still omit them, but training will be much slower.

Our model supports multi-GPU training.
We follow the parallelization paradigm described in [TensorFlow Tutorial][multi-gpu].
In short, if you want to use batch size of 60 (default) but if you have 2 GPUs with 6GB of RAM,
then you initialize each GPU with batch size of 30, and combine the gradients on CPU.
This can be easily done by running:
```
python -m basic.cli --mode train --noload --num_gpus 2 --batch_size 30
```

## Test
To test, run:
```
# run test by specifying the shared json and trained model
export TIMESTAMP=2017-07-10___21-37-44
python -m basic.cli --len_opt --cluster \
--shared_path out/${TIMESTAMP}/basic/00/shared.json \
--load_path out/${TIMESTAMP}/basic/00/save/basic-10000 # the model saved at step 10000

# Test by multi-gpus
python -m basic.cli --len_opt --cluster \
--num_gpus 2 --batch_size 30 \
--zero_threshold 0.02 \ # zero out small weights whose absolute values are <0.02
--shared_path out/${TIMESTAMP}/basic/00/shared.json \
--load_path out/${TIMESTAMP}/basic/00/save/basic-10000 # the model saved at step 10000
```

This command loads the saved model during training and begins testing on the test data.
After the process ends, it prints F1 and EM scores, and also outputs a json file (`$PWD/out/basic/00/answer/test-000000.json`).
Note that the printed scores are not official (our scoring scheme is a bit harsher).
To obtain the official number, use the official evaluator (copied in `squad` folder) and the output json file:
```
python squad/evaluate-v1.1.py \
$HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-000000.json
```

<!--
## Using Pre-trained Model

If you would like to use pre-trained model, it's very easy! 
You can download the model weights [here][save] (make sure that its commit id matches the source code's).
Extract them and put them in `$PWD/out/basic/00/save` directory, with names unchanged.
Then do the testing again, but you need to specify the step # that you are loading from:
```
python -m basic.cli --mode test --batch_size 8 --eval_num_batches 0 --load_step ####
```
-->


## Learning sparse LSTMs
### Learning ISS (hidden states) in LSTMs
```
python -m basic.cli --mode train --len_opt --cluster \
--num_gpus 2 --batch_size 30 \
--input_keep_prob 0.9 \
--load_path out/${TIMESTAMP}/basic/00/save \ # fine-tune baseline, use [--noload] if train from scratch
--structure_wd 0.001 \ # the hyperparameter to make trade-off between sparsity and EM/F1 performance
--group_config groups_hidden100.json # the json to specify ISS structures for LSTMs
```


### Learning sparse LSTMs by L1-norm regularization 
```
# finetuning with L1
python -m basic.cli --mode train --len_opt --cluster --load_path ${HOME}/trained_models/squad/bidaf_adam_baseline/basic-10000 --l1wd 0.0001 --input_keep_prob 0.9 --num_gpus 2 --batch_size 30
```
### Learning to remove columns and rows in the weight matrices of LSTMs by group Lasso regularization 
```
python -m basic.cli --mode train --len_opt --cluster --load_path ${HOME}/trained_models/squad/bidaf_adam_baseline/basic-10000 \
--l1wd 0.0001 --row_col_wd 0.0004 --input_keep_prob 0.9 --num_gpus 2 --batch_size 30
```

### finetuning with zero weights frozen
```
python -m basic.cli --mode train --len_opt --cluster --load_path out//basic/00/save/basic-10000 --freeze_mode element --input_keep_prob 0.9 --init_lr 0.0002 --num_gpus 2 --batch_size 30
```
 
[multi-gpu]: https://www.tensorflow.org/versions/r0.11/tutorials/deep_cnn/index.html#training-a-model-using-multiple-gpu-cards
[squad]: http://stanford-qa.com
[paper]: https://arxiv.org/abs/1611.01603
[worksheet]: https://worksheets.codalab.org/worksheets/0x37a9b8c44f6845c28866267ef941c89d/
[minjoon]: https://seominjoon.github.io
[minjoon-github]: https://github.com/seominjoon
[v0.2.1]: https://github.com/allenai/bi-att-flow/tree/v0.2.1
