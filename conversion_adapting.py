# encoding: UTF-8

from datetime import datetime, timedelta

import pandas as pd
import pickle

user_id_str = 'subsid'
call_time_str = 'time'
cell_id_str = 'cellid'
lac_str = 'lacid'

cell_id_str_report = 'Cell ID'
lac_str_report = 'LAC'

interval_min = 10
nrows = 10000

df = pd.read_csv('./data_routes_csv/petrovsky_11.csv')
report_df = pd.read_csv('./data_routes_csv/report.csv')


def index_of_min(values):
    return min(range(len(values)), key=values.__getitem__)


def get_time(time_str):
    return datetime.strptime(time_str, '%d.%m.%Y %H.%M.%S')


from_time = get_time('24.11.2015 16.00.00')
to_time = get_time('24.11.2015 23.59.00')


def create_aligned_path(data_src, start_time, end_time, interval):
    data_src_cells = data_src[cell_id_str]
    data_src_times = data_src[call_time_str]
    data_src_lacs = data_src[lac_str]

    step = timedelta(minutes=interval)
    cur_time = start_time

    acc = []
    while cur_time <= end_time:
        time_differences = [abs(cur_time - get_time(some_time)) for some_time in data_src_times]
        index = index_of_min(time_differences)
        acc.append((data_src_cells[index], data_src_lacs[index]))
        cur_time += step

    return acc


def between_time(x):
    return to_time > get_time(x) > from_time


df2 = df[df[call_time_str].apply(between_time)]
cells = df2.groupby(user_id_str)[cell_id_str].apply(lambda x: x.tolist())
lacs = df2.groupby(user_id_str)[lac_str].apply(lambda x: x.tolist())
times = df2.groupby(user_id_str)[call_time_str].apply(lambda x: x.tolist())
# Join 2 series to dataframe on user id
data = pd.concat([cells, lacs, times], axis=1).reset_index()

# Add alignded base station sequences
data['route_cell_id'] = data.apply(
    lambda row: create_aligned_path(row, from_time, to_time, interval_min), axis=1)

report_dict = report_df.set_index([cell_id_str_report, lac_str_report]).T.to_dict('list')


def get_coords(path):
    path_ret = []
    for cell_id_lac in path:
        station_info = report_dict[cell_id_lac]
        lon, lat = station_info[0], station_info[1]
        path_ret += [(lon, lat)]

    return path_ret


data['route_coord'] = data['route_cell_id'].apply(get_coords).tolist()

with open('./data_routes_pickle/routes_user_id', 'wb') as f:
    pickle.dump(list(data[user_id_str]), f)

with open('./data_routes_pickle/routes_coord', 'wb') as f:
    pickle.dump(list(data['route_coord']), f)

with open('./data_routes_pickle/routes_cell_id', 'wb') as f:
    pickle.dump(list(data['route_cell_id']), f)

with open('./data_routes_pickle/cell_id_lac_info', 'wb') as f:
    pickle.dump(report_dict, f)
