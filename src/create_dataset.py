import pandas as pd
import numpy as np

from config import Config

Config.RAW_DATASET_FILE_PATH.parent.mkdir(parents= True, exist_ok= True)
Config.CLEANED_DATASET_FILE_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv (Config.RAW_DATASET_FILE_PATH)
#For this use case, we will read the existing data from the original folder.
#If we are reading the data from a link, we can save the raw dataset from git/kaggle/etc. to raw data folder by the below code
        #df = pd.read_csv ("********URL LINK******")
        #df.to_csv (str(Config.ORIGINAL_DATASET_FILE_PATH), index = False)

# The raw data is already cleaned and hence no processing is required.
# We will directly export the dataset from raw_data folder to dataset folder
df.to_csv (str(Config.CLEANED_DATASET_FILE_PATH/"dataset.csv"), index = False)