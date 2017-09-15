# Bi-directional Attention Flow for Machine Comprehension
 
- This is a modified implementation of [Bi-directional Attention Flow for Machine Comprehension][paper] (Seo et al., 2016).
- This is tensorflow v1.1.0 comaptible version. 

## 0. Requirements
#### General
- Python (developed on 3.5.2. Issues have been reported with Python 2!)
- unzip

#### Python Packages
- tensorflow (deep learning library, verified on 1.1.0)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visaulization; if you only train and test, not needed)

## 1. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```

Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
```
python -m squad.prepro
```

## 2. Training BiDAF baseline
Note that the training script save results in the subfolder named as timestamps (e.g. `out/2017-06-28___04-27-50/`). Before running, please:
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


## 3. Test
To test, run:
```
# run test by specifying the shared json and trained model
python -m basic.cli --len_opt --cluster \
--shared_path out/2017-06-28___04-27-50/basic/00/shared.json \
--load_path out/2017-06-28___04-27-50/basic/00/save/basic-10000 # the model saved at step 10000
```

This command loads the saved model during training and begins testing on the test data.
After the process ends, it prints F1 and EM scores, and also outputs a json file (`$PWD/out/basic/00/answer/test-####.json`,
where `####` is the step # that the model was saved).
Note that the printed scores are not official (our scoring scheme is a bit harsher).
To obtain the official number, use the official evaluator (copied in `squad` folder) and the output json file:

```
python squad/evaluate-v1.1.py $HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-####.json
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


## Multi-GPU Training & Testing
Our model supports multi-GPU training.
We follow the parallelization paradigm described in [TensorFlow Tutorial][multi-gpu].
In short, if you want to use batch size of 60 (default) but if you have 2 GPUs with 6GB of RAM,
then you initialize each GPU with batch size of 30, and combine the gradients on CPU.
This can be easily done by running:
```
python -m basic.cli --mode train --noload --num_gpus 2 --batch_size 30

# finetuning with L1
python -m basic.cli --mode train --len_opt --cluster --load_path ${HOME}/trained_models/squad/bidaf_adam_baseline/basic-10000 --l1wd 0.0001 --input_keep_prob 0.9 --num_gpus 2 --batch_size 30

# finetuning with lasso
python -m basic.cli --mode train --len_opt --cluster --load_path ${HOME}/trained_models/squad/bidaf_adam_baseline/basic-10000 \
--l1wd 0.0001 --row_col_wd 0.0004 --input_keep_prob 0.9 --num_gpus 2 --batch_size 30

# learning structure
python -m basic.cli --mode train --len_opt --cluster --load_path ${HOME}/trained_models/squad/bidaf_adam_baseline/basic-10000 --structure_wd 0.002 --input_keep_prob 0.9 --num_gpus 2 --batch_size 30 --group_config groups.json

# finetuning with zero weights frozen
python -m basic.cli --mode train --len_opt --cluster --load_path out//basic/00/save/basic-10000 --freeze_mode element --input_keep_prob 0.9 --init_lr 0.0002 --num_gpus 2 --batch_size 30
```

**Similarly, you can speed up your testing by:**
```
python -m basic.cli --num_gpus 2 --batch_size 30 

# specify the shared json and trained model
export OUTPUT_DIR='out/';
python -m basic.cli --len_opt --cluster --shared_path ${OUTPUT_DIR}/basic/00/shared.json --load_path ${OUTPUT_DIR}/basic/00/save/basic-10000 --num_gpus 2 --batch_size 30 --plot_weights ;
python squad/evaluate-v1.1.py \
$HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-000000.json

```
 

[multi-gpu]: https://www.tensorflow.org/versions/r0.11/tutorials/deep_cnn/index.html#training-a-model-using-multiple-gpu-cards
[squad]: http://stanford-qa.com
[paper]: https://arxiv.org/abs/1611.01603
[worksheet]: https://worksheets.codalab.org/worksheets/0x37a9b8c44f6845c28866267ef941c89d/
[minjoon]: https://seominjoon.github.io
[minjoon-github]: https://github.com/seominjoon
[v0.2.1]: https://github.com/allenai/bi-att-flow/tree/v0.2.1
