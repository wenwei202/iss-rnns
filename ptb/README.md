# Introduction
This is code a modified version of https://www.tensorflow.org/tutorials/recurrent

Use `python ptb_word_lm.py --help` for usage.

# Get ptb data
```
./get_ptd_data.sh
```
# To run
```
python ptb_word_lm.py --model sparselarge \
--data_path simple-examples/data/  \
--config_file structure_grouplasso.json 
```
To finetune a model, we can restore the model by `--restore_path ${HOME}/trained_models/ptb/ptb_large_baseline/`, which points to the path of checkpoint files of `model.ckpt-xxx`.

To freeze zero weights during finetuning, we can use `--freeze_mode element`.

`python ptb_word_lm.py --help` for more help.

# Train two stacked LSTMs with heterogeneous hidden sizes
```
python ptb_word_lm_heter.py \
--model large \
--data_path simple-examples/data/ \
--hidden_size1 373 \
--hidden_size2 315 \
--config_file from_scratch.json 
```

