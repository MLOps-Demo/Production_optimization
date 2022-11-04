import pandas as pd
import json

file_name = "commit_based_anomalies_prod_optimization.csv"

try:
  summary_df = pd.read_csv(file_name)
except FileNotFoundError:
    print("No anomaly summary csv file found in S3, so creating a new one")
    column_nms = ['commit_id','total_observations','anomalies_detected',
    'anomaly_detected_percentage','normal_observations','anomalies_after_treated_by_expert',
    'normal_observations_after_treated_by_expert','treatment_status']
    summary_df = pd.DataFrame(columns = column_nms) 

# Opening JSON file
f = open('anomaly_summary.json')
# returns JSON object as a dictionary
json_anom = json.load(f)

col_dict = dict.fromkeys(column_nms, "NULL")
col_dict['total_observations'] = json_anom['Total observations']
col_dict['anomalies_detected'] = json_anom['Anomalies removed']
col_dict['anomaly_detected_percentage'] = str(json_anom['% Anomaly removed']) + ' %'
col_dict['normal_observations'] = json_anom['Total observations'] - json_anom['Anomalies removed']
col_dict['treatment_status'] = 'FALSE'
# col_dict['total_observations'] = json_anom['Anomalies detected']
# col_dict['total_observations'] = json_anom['% Anomaly identified by model']
#print(col_dict)
anomaly_summary_df = pd.DataFrame([col_dict])
anomaly_summary_df = pd.concat([anomaly_summary_df, summary_df], ignore_index = True)
anomaly_summary_df = anomaly_summary_df.iloc[:9]
anomaly_summary_df.to_csv('commit_based_anomalies_prod_optimization.csv', index = None)
print('Anomaly summary report csv file generated successfully')

