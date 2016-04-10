import pandas as pd
import os

cell_id_str = 'cellid'
lac_str = 'lacid'

cell_id_str_report = 'Cell ID'
lac_str_report = 'LAC'

route_dir = './data_routes_csv/'
convertion_results_dir = './converted/'

# Report convertions
#
# report_df = pd.read_csv('./data_routes_csv/report.csv')
# report_df_lacs = report_df.dropna(subset=[cell_id_str_report, lac_str_report], how='any')
# report_df_lacs = report_df_lacs.drop_duplicates()
#
# report_df_lacs.to_csv('./converted/report_cleared.csv', index=False)
#
# [end] Report convertions

# Route data convertions

report_df = pd.read_csv(route_dir + 'report.csv')
cell_id_in_report = set(tuple(x) for x in report_df[[cell_id_str_report, lac_str_report]].values)

for filename in os.listdir(os.path.join(os.getcwd(), route_dir)):
    if filename.endswith('.csv') and not filename.startswith('report'):
        df = pd.read_csv(route_dir + filename)

        rows_count_intial = df.shape[0]

        df.dropna(subset=[cell_id_str], inplace=True)

        # Drop rows with cell id & LAC not in report
        df['tmp'] = df[[cell_id_str, lac_str]].apply(tuple, axis=1)
        df = df[df['tmp'].isin(cell_id_in_report)]
        del df['tmp']

        print(rows_count_intial - df.shape[0], " rows deleted of ", rows_count_intial)

        # Directory should exist in cwd for this to work
        df.to_csv(convertion_results_dir + filename, index=False)
