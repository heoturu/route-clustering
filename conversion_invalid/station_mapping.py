import pickle
import pandas as pd

cell_id_str_report = 'Cell ID'
lac_str_report = 'LAC'
lon_str_report = 'LON'
lat_str_report = 'LAT'

route_dir = '../data_routes_csv/'
convertion_results_dir = './converted/'

report_df = pd.read_csv(route_dir + 'report.csv')

report_dict = report_df.set_index([cell_id_str_report, lac_str_report]).to_dict(orient='index')

station_map = {}
coord_map = {}

for station in report_dict:
    coords = report_dict[station]['LON'], report_dict[station]['LAT']

    if coords not in coord_map:
        coord_map[coords] = station

    station_map[station] = coord_map[coords]

with open('../data_routes_pickle/station_map', 'wb') as f:
    pickle.dump(station_map, f)
