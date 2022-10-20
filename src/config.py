from pathlib import Path

class Config:
    DATA_PATH = Path("./data")
    RAW_DATASET_FILE_PATH = DATA_PATH/"raw_dataset"/ "clean_data.csv"
    CLEANED_DATASET_FILE_PATH = DATA_PATH/ "cleaned_data"
    ANOMALY_TREATED_DATAS_FILE_PATH = DATA_PATH/"anomaly_treated" 
    FEATURES_PATH = DATA_PATH/ "features"
    MODELS_PATH = DATA_PATH/ "models"
    PLOTS_PATH = DATA_PATH/"plots_figs"
    PREDICTION_FILE_PATH = DATA_PATH/"predictions"
    #METRICS_FILE_PATH = DATA_PATH/"metrics.json"
