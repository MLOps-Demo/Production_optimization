import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import warnings
warnings.filterwarnings("ignore")
import json

#import dimnesionality reduction algorithms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#import anomaly detection models
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.cblof import CBLOF

from config import Config
Config.ANOMALY_TREATED_DATAS_FILE_PATH.mkdir(parents = True, exist_ok = True)
Config.PLOTS_PATH.mkdir(parents = True, exist_ok = True)

params = yaml.safe_load(open("params.yaml"))["anomaly_detect"]
seed = params["seed"]
contamination = params["anomaly_contamination"]
neighbours = params["neighbours_knn"]
clusters = params["clusters_cblof"]
nu = params["nu_ocsvm"]
tol = params["tol_ocsvm"]

df = pd.read_csv(Config.CLEANED_DATASET_FILE_PATH/"dataset.csv")
feature_names = df.drop(['NewDateTime'], axis=1).columns.tolist()    #excluding the first column 'NewDateTime' and choose rest of features

def anomaly_detect(X, feat_names):
    global anomaly_point_index
    
    classifiers = {
    'KNN' :  KNN(contamination = contamination, n_neighbors=neighbours, n_jobs=-1),
    'LOF'   : LOF(contamination= contamination, n_neighbors=neighbours, n_jobs=-1),
    'IForest' : IForest(contamination= contamination, n_jobs=-1, random_state = seed),
    'OCSVM' : OCSVM(contamination = contamination, gamma = 'scale', nu = nu, tol = tol),
    'CBLOF' : CBLOF(contamination = contamination, random_state = seed, n_jobs = -1, n_clusters = clusters)
    }
    
    original_len_df = len(X)
    original_shape = X.shape
    print ("Anomaly contamination considerd for dataset =", contamination*100, "%")
    for i, (clf_name,clf) in enumerate(classifiers.items()) :      
        # fit the dataset to the model
        print ('Running anomaly detector Model {}: {}'.format(i+1,clf_name))
        clf.fit(df[feature_names])
        #Predict the outlier labels
        y_pred = clf.labels_
        #add the outlier label column to the dataframe
        new_col = 'y_pred_' + clf_name
        X[new_col] = y_pred
    print('Anomaly detection model execution completed')
    # Creating an average of the prediction labels of 5 models
    X['anomaly_wt'] = (df.y_pred_KNN + df.y_pred_LOF + df.y_pred_IForest + df.y_pred_OCSVM + df.y_pred_CBLOF) / 5
    # Unique values for the above avrage are [0.0, 0.2, 0.4, 0.6 and 0.8]. 
    # Considering only 0.4 and above as anomaly when atleast 2 models predicted anomaly of an data point.
    X['anomaly_pred'] = ""
    X['anomaly_pred'][X.anomaly_wt >= 0.4] = 'Anomaly'
    X['anomaly_pred'][X.anomaly_wt < 0.4] = 'Normal'
    #storing the index of the anomalous data point to visualize
    anomaly_point_index = list(X.loc[X['anomaly_pred'] == 'Anomaly'].index)
    print('\nResults based on combining the predictions of the 5 models: \n -Index of anomalous datapoints : \n',
          anomaly_point_index)
    
    #Filtering the Anomalous data points from the dataframe and dropping the newly added model label predicted columns
    anomalous_df = X[X.anomaly_pred == 'Anomaly'].copy()
    anomalous_df.drop(['y_pred_KNN','y_pred_LOF','y_pred_IForest','y_pred_OCSVM','y_pred_CBLOF',
             'anomaly_wt','anomaly_pred'], axis=1, inplace=True)
    X = X[X.anomaly_pred == 'Normal']
    X.drop(['y_pred_KNN','y_pred_LOF','y_pred_IForest','y_pred_OCSVM','y_pred_CBLOF',
             'anomaly_wt','anomaly_pred'], axis=1, inplace=True)

    final_len_df = len(X)
    final_shape = X.shape
    
    print("\n-Original shape of dataframe: ", original_shape , "\n-Final shape of dataframe: " , final_shape)
    print("-Total anomalies detected and removed = ", original_len_df - final_len_df)
    anomaly_samples_removed = original_len_df - final_len_df
    anomalies_percent = "{:.2f}".format((anomaly_samples_removed/original_len_df)*100) + ' %'
    print("-Percentage of anomalies removed =", anomalies_percent)
    #path to save the anomaly treated dataset
    normal_path = str(Config.ANOMALY_TREATED_DATAS_FILE_PATH/ "anomaly_treated_data.csv")
    X.to_csv(normal_path, index = None)
    anomaly_path = str(Config.ANOMALY_TREATED_DATAS_FILE_PATH/ "anomalous_data.csv")
    anomalous_df.to_csv(anomaly_path, index = None)
    print("\nExported the Anomaly treated and anomalous dataset separately to csv file in the folder -> '{}'".format(str(Config.ANOMALY_TREATED_DATAS_FILE_PATH)))
    
    with open("anomaly_summary.json", "w") as fd:
        json.dump({"Total observations": original_len_df,
        "% Anomaly removed": round((anomaly_samples_removed/original_len_df)*100, 2),
        "Anomalies removed": anomaly_samples_removed }, fd, indent=4)

    return

def anomaly_plots(X, feat_names, anomaly_index_list):
    sns.set_theme(style="whitegrid")
    
    dim_reductors = {
    'PCA' :  PCA(2),
    'TSNE'   : TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=seed)
    }
      
    for i, (model_name,model) in enumerate(dim_reductors.items()) :
        print("Running dimensionality reduction model {} to generate 2 dimensional Data...".format(model_name))
        # fit the dataset to the model and save as a dataframe
        result = pd.DataFrame(model.fit_transform(X[feat_names]))
        
        if model_name == 'PCA':
            print("Generating 2D anomaly plot on PCA components...")
            points_color = 'green'
            plt_title = "Anomaly Detection (by reducing dataset dimension to 2D using PCA)"
            x_label = "PCA Component #1"
            y_label = "PCA Component #2"
            path = str(Config.PLOTS_PATH/ "anomaly_detect_plt_pca.png")
        else :
            print("Generating 2D anomaly plot on t-SNE components...")
            points_color = 'blue'
            plt_title = "Anomaly Detection (by reducing dataset dimension to 2D using t-SNE)"
            x_label = "t-SNE Component #1"
            y_label = "t-SNE Component #2"
            path = str(Config.PLOTS_PATH/ "anomaly_detect_plt_tsne.png")
        
        plt.figure(figsize=(8,8))
        a = plt.scatter(result[0], result[1],c = points_color, label="Normal")
        a = plt.scatter(result.iloc[anomaly_point_index,0], result.iloc[anomaly_point_index,1] ,
                        c = 'yellow', edgecolor="red", label="Anomaly")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper right")
        plt.title(plt_title)
        plt.tight_layout()
        plt.savefig(path, dpi = 150)
        print("Plot saved")
        folderpath = str(Config.PLOTS_PATH)    
    print("\nAnomaly plots on PCA and t-SNE components saved in .png format in the folder -> '{}'".format(folderpath))
    return

if __name__ == '__main__':
    anomaly_detect(df, feature_names)
    anomaly_plots(df, feature_names, anomaly_point_index)