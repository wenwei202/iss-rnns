# Introduction
This is code from https://www.tensorflow.org/tutorials/recurrent

# Get ptb data
```
./get_ptd_data.sh
```
# To run
```
python ptb_word_lm.py --model sparselarge --data_path simple-examples/data/  --restore_path ${HOME}/trained_models/ptb/ptb_large_baseline/ --config_file structure_grouplasso.json 
```
