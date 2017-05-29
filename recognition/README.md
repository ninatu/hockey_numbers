# TRAINING and TESTING neural network model

Dataset tools:
usage: `python dataset_tools.py [-h] {merge,balance,crop,sample,split,svhn} ...`

Train/test neural networks:

1. `cd keras`
2. Modify file constants.py. Please specify MODEL_DATA_DIR, DATA_FOLDER. (DIR_NOT_NUMBER_CROP, DIR_NOT_NUMBER_HALF only for binary classification "number"/"not number")
3. `python main.py [-h] {prepare,train,evaluate,predict} ...`
