import os
import pickle
import pandas as pd

cell_id_str = 'cellid'
lac_str = 'lacid'

cell_id_str_report = 'Cell ID'
lac_str_report = 'LAC'
lon_str_report = 'LON'
lat_str_report = 'LAT'

tuple_str = 'tuple'
new_cell_id_lac_str = 'new_cell_id_lac'

route_dir = '../data_routes_csv/'
convertion_results_dir = '../converted/'

# Route data convertions

report_df = pd.read_csv(route_dir + 'report.csv')
report_cleared = pd.read_csv(route_dir + 'report_cleared.csv')
cell_id_in_report = set(tuple(x) for x in report_df[[cell_id_str_report, lac_str_report]].values)

with open('../data_routes_pickle/station_map', 'rb') as f:
    station_map = pickle.load(f)


for filename in os.listdir(os.path.join(os.getcwd(), route_dir)):
    if filename.endswith('.csv') and not filename.startswith('report'):
        df = pd.read_csv(route_dir + filename)

        rows_count_intial = df.shape[0]

        df.dropna(subset=[cell_id_str], inplace=True)

        # Drop rows with cell id & LAC not in report
        df[tuple_str] = df[[cell_id_str, lac_str]].apply(tuple, axis=1)
        df = df[df[tuple_str].isin(cell_id_in_report)]
        del df[tuple_str]

        # Use only selected report base stations

        df[new_cell_id_lac_str] = \
            df[[cell_id_str, lac_str]].apply(lambda cell_id_lac: station_map[tuple(cell_id_lac)], axis=1)

        del df[cell_id_str]
        del df[lac_str]

        df[[cell_id_str, lac_str]] = df[new_cell_id_lac_str].apply(pd.Series)

        del df[new_cell_id_lac_str]

        # [end] Use only selected report base stations

        print(rows_count_intial - df.shape[0], " rows deleted of ", rows_count_intial)

        # Directory should exist in cwd for this to work
        df.to_csv(convertion_results_dir + filename, index=False)
