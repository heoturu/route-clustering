from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

# xl = pd.ExcelFile("m1.xlsx")
# df = xl.parse('Sheet1')

df = pd.read_csv('petrovsky_11.csv', nrows=50000)

report_df = pd.read_csv('report.csv')


def index_of_min(values):
    return min(range(len(values)), key=values.__getitem__)


def get_time(x):
    return datetime.strptime(x, "%d.%m.%Y %H.%M.%S")


from_time = get_time("24.11.2015 11.00.00")
to_time = get_time("24.11.2015 15.00.00")


def create_aligned_path(data, start_time, end_time, interval):
    lacs, times = data["cellid"], data["time"]
    step = timedelta(minutes=interval)
    current_time = start_time
    acc = []
    while current_time <= end_time:
        time_differences = [abs(current_time - get_time(some_time)) for some_time in times]
        index = index_of_min(time_differences)
        acc.append(lacs[index])
        current_time += step
    return acc


def between_time(x):
    return to_time > get_time(x) > from_time

# Choose only events in some time period
# Group people by code - unique number for every user
# Collect all LACs (base stations) for every user in a list

df1 = df[df['time'].apply(between_time)]
df2 = df1.dropna(subset=['cellid'])

lacs = df2.groupby('subsid')['cellid'].apply(lambda x: x.tolist())
# The same with all times for every LAC
times = df2.groupby('subsid')['time'].apply(lambda x: x.tolist())

# Join 2 series to dataframe on Code
data = pd.concat([lacs, times], axis=1).reset_index()

# Add alignded LACs sequences
data['path'] = data.apply(lambda x: create_aligned_path(x, from_time, to_time, 10), axis=1)

report_df.drop_duplicates(subset=['Cell ID'], inplace=True)

report_dict = report_df.set_index('Cell ID').T.to_dict('list')


def get_coords(path):
    path_ret = []
    for lac in path:
        lac_info = report_dict[lac]
        lon, lat = lac_info[1], lac_info[2]
        path_ret += [lon, lat]

    return path_ret


def split_timespan(chunk_count):
    delta = (to_time - from_time) / chunk_count
    return [to_time + delta * idx for idx in range(chunk_count)]


coord_list = data['path'].apply(get_coords).tolist()

rng = np.random.RandomState(0)
kmeans = MiniBatchKMeans(n_clusters=2, random_state=rng, verbose=True)

kmeans.partial_fit(coord_list)

for i, patch in enumerate(kmeans.cluster_centers_):
    print("Cluster #" + str(i) + ": " + str(patch))
