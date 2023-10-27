import os 

# path to folder with data 
DATASET_FOLDER = os.path.expanduser("dataset")
trainDirectory = os.path.join(DATASET_FOLDER, "train")
testDirectory = os.path.join(DATASET_FOLDER, "test")

# training and val split 
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10

# Batch size, total N of epochs, learning rate 
BATCH_SIZE = 16
NUM_OF_EPOCHS = 50
LR = 1e-1