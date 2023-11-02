import os 

# path to folder with data 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(ROOT_DIR, "dataset")
trainDirectory = os.path.join(DATASET_FOLDER, "train")
testDirectory = os.path.join(DATASET_FOLDER, "test")
