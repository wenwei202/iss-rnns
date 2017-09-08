# Introduction
This is code from https://www.tensorflow.org/tutorials/recurrent

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
