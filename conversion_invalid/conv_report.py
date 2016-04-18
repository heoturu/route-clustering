import pickle
import pandas as pd

cell_id_str_report = 'Cell ID'
lac_str_report = 'LAC'
lon_str_report = 'LON'
lat_str_report = 'LAT'

route_dir = '../data_routes_csv/'
convertion_results_dir = '../converted/'

report_df = pd.read_csv(route_dir + 'report.csv')

with open('../data_routes_pickle/station_map', 'rb') as f:
    station_map = pickle.load(f)

report_df_lacs = report_df.dropna(subset=[cell_id_str_report, lac_str_report], how='any')
report_df_lacs = report_df_lacs.drop_duplicates()
report_df_lacs = report_df_lacs.drop_duplicates(subset=[lat_str_report, lon_str_report])

report_df['tuple'] = report_df[[cell_id_str_report, lac_str_report]].apply(tuple, axis=1)
report_df['mapped'] = \
    report_df[[cell_id_str_report, lac_str_report]]\
    .apply(lambda cell_id_lac: station_map[tuple(cell_id_lac)], axis=1)

report_df_lacs = report_df[report_df['tuple'] == report_df['mapped']]

del(report_df_lacs['tuple'])
del(report_df_lacs['mapped'])

report_df_lacs.to_csv(convertion_results_dir + 'report_cleared.csv', index=False)
