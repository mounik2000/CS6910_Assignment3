# CS6910_Assignment3
Use recurrent neural networks to build a transliteration system.

Assignment link : https://wandb.ai/miteshk/assignments/reports/Assignment-3--Vmlldzo0NjQwMDc

wandb report link : https://wandb.ai/mounik2000/Assignment%203%20Attention%20Added/reports/Assignment-3--Vmlldzo2ODY4MTU?accessToken=nqthro5qcyj066t7xunxs8cdcl9vx5gr30r3t28lsf618leudj1iyma95emxfk1j


## Instructions for running Translitertion.py: ##

If you plan to run on colab, upload all the input files (tsv files) and fonts and copy this python file code in colab notebook cell.

If you plan on running your system, all the input files and fonts present in this repo along with the .py file should be in the same folder. Safely, you can clone this repo and run terminal from that local folder.

Change entities in line no - 449 to your wandb username and project. Make sure you have all required libraries installed.

Give appropriate hyperparameter values in the sweep config. 'attention' parameter to be set to 1 if you are using attention. Otherwise it may be set to 0.

It is _**highly recommended**_ to run the code in presence of a GPU with faster RAM for a shorter runtime and to prevent out of memory errors.


### To run the script ###

Run the command - _**python Translitertion.py**_ (Using python 3.x only) in your terminal or execute the cell in the notebook (if colab or jupyter is used.)

## Instructions for running test_time.py: ##

If you plan to run on colab, upload all the input files (tsv files) and fonts and copy this python file code in colab notebook cell.

If you plan on running your system, as all the input files and fonts present in this repo along with the .py file should be in the same folder. Safely, you can clone this repo and run terminal from that local folder.

Change entities in line no - 583 to your wandb username and project. Make sure you have all required libraries installed.

Give appropriate best hyperparameter values you have found in the sweep config. 'attention' hyperparameter to be set to 1 if you are using attention. Otherwise it may be set to 0.

It is _**highly recommended**_ to run the code in presence of a GPU with faster RAM for a shorter runtime and to prevent out of memory errors.


### To run the script ###

Run the command - _**python test_time.py**_ (Using python 3.x only) in your terminal or execute the cell in the notebook (if colab or jupyter is used.)

## Acknowledgements ##

The tutorials in the following links are referred to while writing the code.

1. https://keras.io/examples/nlp/lstm_seq2seq/
2. https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
3. https://wandb.ai/mathisfederico/wandb_features/reports/Better-visualizations-for-classification-problems--VmlldzoxMjYyMzQ
4. https://stackoverflow.com/questions/48302284/how-to-color-text-by-character-in-python
5. https://www.tensorflow.org/tutorials/text/nmt_with_attention
