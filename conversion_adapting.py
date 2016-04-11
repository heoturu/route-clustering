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

nrows = 10000
interval_min = 10
measure_approx_boundary = interval_min  # TODO Seems better to use smth like interval_min / 3

dist_measure = 'standard'

df = pd.read_csv('./data_routes_csv/user_49454_bug.csv')

with open('./data_routes_pickle/cell_id_lac_info', 'rb') as f:
    report_dict = pickle.load(f)


def index_of_min(values):
    return min(range(len(values)), key=values.__getitem__)


def get_time(time_str):
    return datetime.strptime(time_str, '%d.%m.%Y %H.%M.%S')


from_time = get_time('24.11.2015 16.00.00')
to_time = get_time('24.11.2015 23.59.00')


def create_aligned_path(data_src, start_time, end_time, measure):
    def measure_standard():
        return data_src_cells[mindiff_idx], data_src_lacs[mindiff_idx]

    def measure_approx():
        if (abs(time_diffs[mindiff_idx]) <= step or
                time_diffs_neg.empty or
                time_diffs_pos.empty):
            station_info = report_dict[(data_src_cells[mindiff_idx], data_src_lacs[mindiff_idx])]
            return station_info[0], station_info[1]
        else:
            station1_info = report_dict[(data_src_cells[max_neg_idx], data_src_lacs[max_neg_idx])]
            station2_info = report_dict[(data_src_cells[min_pos_idx], data_src_lacs[min_pos_idx])]

            coeff = (cur_time - data_src_times[max_neg_idx]) / \
                    (data_src_times[min_pos_idx] - data_src_times[max_neg_idx])

            approx_coords = \
                station1_info[0] + coeff * (station2_info[0] - station1_info[0]), \
                station1_info[1] + coeff * (station2_info[1] - station1_info[1]),

            return approx_coords

    data_src_cells = data_src[cell_id_str]
    data_src_times = pd.Series(data_src[call_time_str]).apply(lambda time_str: get_time(time_str))
    data_src_lacs = data_src[lac_str]

    zero_time = timedelta()
    step = timedelta(minutes=interval_min)
    cur_time = start_time

    acc = []
    while cur_time <= end_time:
        time_diffs = data_src_times.apply(lambda time: time - cur_time)

        time_diffs_neg = time_diffs[time_diffs < zero_time]
        time_diffs_pos = time_diffs[time_diffs >= zero_time]

        if time_diffs_neg.empty:
            mindiff_idx = time_diffs_pos.idxmin()
        elif time_diffs_pos.empty:
            mindiff_idx = time_diffs_neg.idxmax()
        else:
            max_neg_idx = time_diffs_neg.idxmax()
            min_pos_idx = time_diffs_pos.idxmin()

            if abs(time_diffs[max_neg_idx]) < time_diffs[min_pos_idx]:
                mindiff_idx = max_neg_idx
            else:
                mindiff_idx = min_pos_idx

        if measure == 'standard':
            used_measure = measure_standard
        elif measure == 'approx':
            used_measure = measure_approx
        else:
            raise Exception('Unknown station location measure')

        acc.append(used_measure())
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

if dist_measure == 'standard':
    data['route_cell_id'] = \
        data.apply(lambda row: create_aligned_path(row, from_time, to_time, dist_measure), axis=1)


    def get_coords(path):
        path_ret = []
        for cell_id_lac in path:
            station_info = report_dict[cell_id_lac]
            lon, lat = station_info[0], station_info[1]
            path_ret += [(lon, lat)]

        return path_ret


    data['route_coord'] = data['route_cell_id'].apply(get_coords).tolist()

    with open('./data_routes_pickle/routes_cell_id', 'wb') as f:
        pickle.dump(list(data['route_cell_id']), f)
elif dist_measure == 'approx':
    data['route_coord'] = \
        data.apply(lambda row: create_aligned_path(row, from_time, to_time, dist_measure), axis=1)
else:
    raise Exception('Unknown station location measure defined')

with open('./data_routes_pickle/routes_user_id', 'wb') as f:
    pickle.dump(list(data[user_id_str]), f)

with open('./data_routes_pickle/routes_coord', 'wb') as f:
    pickle.dump(list(data['route_coord']), f)

# Creating report_dict
#
# report_df = pd.read_csv('./data_routes_csv/report.csv')
# report_dict = report_df.set_index([cell_id_str_report, lac_str_report]).T.to_dict('list')
#
# with open('./data_routes_pickle/cell_id_lac_info', 'wb') as f:
#     pickle.dump(report_dict, f)
#
# [end] Creating report_dict
