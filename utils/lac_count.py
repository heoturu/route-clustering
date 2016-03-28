from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

df = pd.read_csv('../old_petrovsky_11.csv')
report_df = pd.read_csv('../report.csv')

df_lacs = df.drop_duplicates(subset=['LAC'])
df_lacs = df_lacs.dropna(subset=['LAC'])

report_df_lacs = report_df.drop_duplicates(subset=['LAC'])
report_df_lacs = report_df_lacs.dropna(subset=['LAC'])

# print(df_lacs['LAC'])
# print(report_df_lacs['LAC'])

print("Route spreadsheet: " + str(len(df_lacs)))
print("Report spreadsheet: " + str(len(report_df_lacs)))

lac_dict = {}

# Simple counting for LACs that are in only one (any of the two) df

# for _, row in df_lacs.iterrows():
#     if row['LAC'] not in lac_dict:
#         lac_dict[row['LAC']] = 1
#     else:
#         lac_dict[row['LAC']] += 1
#
# for _, row in report_df_lacs.iterrows():
#     if row['LAC'] not in lac_dict:
#         lac_dict[row['LAC']] = 1
#     else:
#         lac_dict[row['LAC']] += 1
#
# for lac in lac_dict:
#     if lac_dict[lac] == 2:
#         pass
#     elif lac_dict[lac] == 1:
#         print(lac)

# [end] Simple counting

for _, row in df_lacs.iterrows():
    if row['LAC'] not in lac_dict:
        lac_dict[row['LAC']] = 1

for _, row in report_df_lacs.iterrows():
    if row['LAC'] not in lac_dict:
        lac_dict[row['LAC']] = 0
    elif lac_dict[row['LAC']] != 0:
        lac_dict[row['LAC']] += 1

lacs_only_file = []
lacs_only_report = []

for lac in lac_dict:
    if lac_dict[lac] == 1:
        lacs_only_file.append(lac)
    elif lac_dict[lac] == 0:
        lacs_only_report.append(lac)

print(lacs_only_report)
print(lacs_only_file)

activities_only_file_count = str(len(df[df['LAC'].isin(lacs_only_file)]))
activities_correct_count = str(len(df[df['LAC'].apply(lambda x: x not in lacs_only_file)]))

print("Number of LACs in file but not in report = " + activities_only_file_count)
print("Number of LACs in both file & report = " + activities_correct_count)

# print(df['LAC'])
