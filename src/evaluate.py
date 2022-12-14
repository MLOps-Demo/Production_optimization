import sklearn.metrics as metrics
import pickle
import json
import numpy as np
import pandas as pd
from learning_curves import deviance, feature_importance
from data_split import load_data
import warnings
warnings.filterwarnings("ignore")

from config import Config
Config.PREDICTION_FILE_PATH.mkdir(parents = True, exist_ok= True)

def evaluate():
    print("Model Evaluation")
    x_test = np.load(str(Config.FEATURES_PATH/'x_test.npy'))
    y_test = np.load(str(Config.FEATURES_PATH/'y_test.npy'))

    scaling_model = pickle.load(open(str(Config.MODELS_PATH/ "scaling_model.pkl"), "rb"))
    x_te_scale = scaling_model.transform(x_test)
    print("done")

    model = pickle.load(open(str(Config.MODELS_PATH/ "gbrt_model.pkl"), "rb"))

    fig2 = deviance(model, x_te_scale, y_test)
    fig2.savefig(str(Config.PLOTS_PATH/ "deviance.png"), dpi = 150)

    _, cols = load_data()
    fig3 = feature_importance(model, cols, x_te_scale, y_test)
    fig3.savefig(str(Config.PLOTS_PATH/ "feature_importance.png"), dpi = 150)

    predictions = model.predict(x_te_scale)
    prediction_csv = pd.DataFrame({"target_labels": y_test,
                                   "predicted_labels": predictions})
    prediction_csv.to_csv(str(Config.PREDICTION_FILE_PATH/ "prediction.csv"), index=False)

    mse = metrics.mean_squared_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)

    with open("scores.json", "w") as fd:
        json.dump({"mse": mse, "r2": r2}, fd, indent=4)


if __name__ == '__main__':
    evaluate()
