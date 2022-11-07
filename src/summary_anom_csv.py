import pandas as pd
import json

file_name = "commit_based_anomalies_prod_optimization.csv"

# Defining the column names for the dictionary/ dataframe creation
column_nms = ['commit_id','branch_name','total_observations','anomalies_detected',
'anomaly_detected_percentage','normal_observations','anomalies_after_treated_by_expert',
'normal_observations_after_treated_by_expert','treatment_status']

try:
  summary_df = pd.read_csv(file_name)
except FileNotFoundError:
    print("No anomaly summary csv file found in S3, so creating a new one")
    summary_df = pd.DataFrame(columns = column_nms) 

# Opening Commit ID JSON file generated from Github actions workflow
f = open('commit_id.json')
# returns JSON object as a dictionary
json_commit_id = json.load(f)

# Opening anomaly summary JSON file
f = open('anomaly_summary.json')
# returns JSON object as a dictionary
json_anom = json.load(f)

#Append the values from the json file to new dictionary
col_dict = dict.fromkeys(column_nms, "NULL")
col_dict['commit_id'] = json_commit_id['Sha']
col_dict['branch_name'] = json_commit_id['Branch']
col_dict['total_observations'] = json_anom['Total observations']
col_dict['anomalies_detected'] = json_anom['Anomalies removed']
col_dict['anomaly_detected_percentage'] = str(json_anom['% Anomaly removed']) + ' %'
col_dict['normal_observations'] = json_anom['Total observations'] - json_anom['Anomalies removed']
col_dict['treatment_status'] = 'FALSE'

#Create new dataframe from the newly created dictionary and concat with data from AWS S3 (commit summary history)
anomaly_summary_df = pd.DataFrame([col_dict])
anomaly_summary_df = pd.concat([anomaly_summary_df, summary_df], ignore_index = True)
anomaly_summary_df = anomaly_summary_df.iloc[:9]
anomaly_summary_df.to_csv('commit_based_anomalies_prod_optimization.csv', index = None)
print('Anomaly summary report csv file generated successfully')

