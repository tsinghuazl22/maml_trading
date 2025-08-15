import pathlib



import pandas as pd
import datetime
import os












TRAINING_DATA_FILE = "data/IF300.csv"

now = datetime.datetime.now().strftime('%Y%m%d')



TRAINED_MODEL_DIR = f"trained_models"
TENSORBOARD_DIR = f"tensorboard"
RESULT_DIR = f"results"
PRE_TRAIN_DIR = f"pre_train_model"




if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

TESTING_DATA_FILE = "test.csv"


