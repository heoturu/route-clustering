import os
import pandas as pd


def get_inconsistent_lacs(df_src, report_df_lacs_src):
    df_lacs = df_src.drop_duplicates(subset=[lac_col_name])
    df_lacs = df_lacs.dropna(subset=[lac_col_name])

    lac_dict = {}

    for _, row in df_lacs.iterrows():
        if row[lac_col_name] not in lac_dict:
            lac_dict[row[lac_col_name]] = 1

    for _, row in report_df_lacs_src.iterrows():
        if row[lac_report_name] not in lac_dict:
            lac_dict[row[lac_report_name]] = 0
        elif lac_dict[row[lac_report_name]] != 0:
            lac_dict[row[lac_report_name]] += 1

    lacs_only_file = []
    lacs_only_report = []

    for lac in lac_dict:
        if lac_dict[lac] == 1:
            lacs_only_file.append(lac)
        elif lac_dict[lac] == 0:
            lacs_only_report.append(lac)

    return lacs_only_file


def remove_lacs_invalid(df_src, lacs_filter_out):
    df_dropna = df_src.dropna(subset=[lac_col_name])
    return df_dropna[df_dropna[lac_col_name].apply(lambda x: x not in lacs_filter_out)]


lac_col_name = 'lacid'
lac_report_name = 'LAC'

report_df = pd.read_csv('../report.csv')
report_df_lacs = report_df.drop_duplicates(subset=[lac_report_name])
report_df_lacs = report_df_lacs.dropna(subset=[lac_report_name])

for filename in os.listdir(os.getcwd()):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        lacs_to_preserve = get_inconsistent_lacs(df, report_df_lacs)
        df = remove_lacs_invalid(df, lacs_to_preserve)

        # "converted" directory should exist in cwd for this to work
        df.to_csv('./converted/' + filename, index=False)
