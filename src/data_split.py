"""
Split raw data into training and testing data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

from config import Config

Config.FEATURES_PATH.mkdir(parents = True, exist_ok = True)

params = yaml.safe_load(open("params.yaml"))["data-split"]
split = params["split"]
seed = params["seed"]


def load_data():
    print("Loading data from given folder")
    df = pd.read_csv(str(Config.ANOMALY_TREATED_DATAS_FILE_PATH/ 'anomaly_treated_data.csv')).set_index('NewDateTime')
    cols = df.columns.tolist()
    print("done")
    return df, cols


def data_split(df):
    array = df.values

    x = array[:, :-1]
    y = array[:, -1]

    print("Splitting data into train and test")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=seed)
    print("done")

    np.save(str(Config.FEATURES_PATH/'x_train'), x_train)
    np.save(str(Config.FEATURES_PATH/'x_test'), x_test)
    np.save(str(Config.FEATURES_PATH/'y_train'), y_train)
    np.save(str(Config.FEATURES_PATH/'y_test'), y_test)
    print("Saved data into processed_data folder")


if __name__ == '__main__':
    data, _ = load_data()
    data_split(data)
