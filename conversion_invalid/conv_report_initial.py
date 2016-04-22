import pandas as pd

cell_id_str_report = 'Cell ID'
lac_str_report = 'LAC'

route_dir = '../data_routes_csv/'
convertion_results_dir = '../converted/'

report_df = pd.read_csv(route_dir + 'report.csv')

report_df = report_df.dropna(subset=[cell_id_str_report, lac_str_report], how='any')
report_df = report_df.drop_duplicates()

report_df.to_csv(convertion_results_dir + 'report_cleared.csv', index=False)
