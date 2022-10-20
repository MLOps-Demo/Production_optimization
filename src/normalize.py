"""
Standard Scaling the raw data
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

from config import Config

Config.MODELS_PATH.mkdir(parents = True, exist_ok = True)


def normalize():
    print("Normalizing the data")

    print("Loading split data")
    x_train = np.load(str(Config.FEATURES_PATH/'x_train.npy'))
    x_test = np.load(str(Config.FEATURES_PATH/'x_test.npy'))
    print("done")

    print("Scaling data with Standard Scaler")
    scaling = StandardScaler()
    scaling.fit(x_train)
    print("done")

    with open(str(Config.MODELS_PATH/ "scaling_model.pkl"), "wb") as x_f:
        pickle.dump(scaling, x_f)


if __name__ == '__main__':
    normalize()
