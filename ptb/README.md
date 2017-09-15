# Introduction
This is code a modified version of https://www.tensorflow.org/tutorials/recurrent

By default, LSTMs have hidden sizes of 1500.
# Get ptb data
```
./get_ptd_data.sh
```

# Usage
By default, trained models are saved similarly as `/tmp/2017-06-12___22-48-13/`, where foldername is the time when training started.

To finetune a model, we can restore the model by `--restore_path /tmp/2017-06-12___22-48-13/`, which points to the path of checkpoint files of `model.ckpt-xxx`.

To freeze zero weights during finetuning, we can use `--freeze_mode element`.

Use `python ptb_word_lm.py --help` for more usage.
# To run
## learning non-structurally sparse LSTMs with L1-norm regularization
Finetuning trained model by L1-norm regularization
```
python ptb_word_lm.py --model sparselarge \
--data_path simple-examples/data/ \
--restore_path  /tmp/ptb_large_baseline/  \
--config_file l1.json
```
Weight decay of L1-norm, dropout, etc., are configured in `l1.json`.

## learning ISS with group Lasso regularization
```
python ptb_word_lm.py --model sparselarge \
--data_path simple-examples/data/  \
--config_file structure_grouplasso.json 
```

`structure_grouplasso.json` (the default json to learn ISS from scratch), where `global_decay` is the hyperparameter (lambda) to make trade-off between perlexity and sparsity.

## Evaluate and display weight matrices of trained model
```
python ptb_word_lm.py \
--model validtestlarge \
--data_path simple-examples/data/ \
--display_weights True \
--config_file l1.json \
--restore_path /tmp/2017-06-12___22-48-13/
```

## Directly design two stacked LSTMs with specified hidden sizes and train from scratch
```
python ptb_word_lm_heter.py \
--model large \
--data_path simple-examples/data/ \
--hidden_size1 373 \
--hidden_size2 315 \
--config_file from_scratch.json 
```

