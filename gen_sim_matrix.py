import pickle
from itertools import product

import math
import numpy as np

import utils_map
from utils_dist import *


with open('data_routes_pickle/routes_coord_any_3_contains_nocycles_angle_1.2_1k', 'rb') as f:
    routes_coord = pickle.load(f)

coord_list = routes_coord
# For 'norm' metric
# coord_list = [[num for coords in route for num in coords] for route in routes_coord]


def compute_distance(route1, route2, metric):
    if metric == 'norm':
        return np.linalg.norm(np.array(route1) - np.array(route2), ord=1)
    elif metric == 'sim_points':
        total_common_count = 0
        route1_set = set(route1)
        route2_set = set(route2)

        most_sim_points = min(product(route1_set, route2_set), key=dist_vinc)

        while (vincenty(most_sim_points[0], most_sim_points[1]).miles < threshold_dist_miles and
               min(len(route1_set), len(route2_set)) > 1):
            total_common_count += 1

            route1_set.remove(most_sim_points[0])
            route2_set.remove(most_sim_points[1])

            most_sim_points = min(product(route1_set, route2_set), key=dist_vinc_pair)

        return (min(len(route1), len(route2))) - total_common_count
    elif metric == 'sim_segments':
        # Remove (only consecutive!) duplicates
        route1_unique = route1
        route2_unique = route2

        total_sim_segment_count = 0
        for i in range(len(route1_unique) - 1):
            for j in range(len(route2_unique) - 1):
                # if similar_segments((route1_unique[i], route1_unique[i + 1]),
                #                     (route2_unique[j], route2_unique[j + 1])):
                #     total_sim_segment_count += \
                #         (dist_to_stadium(route1_unique[i]) + dist_to_stadium(route1_unique[i + 1]) / 2 +
                #          dist_to_stadium(route2_unique[j]) + dist_to_stadium(route2_unique[j + 1]) / 2)

                segment_dist_sum_cur = segment_dist_sum((route1_unique[i], route1_unique[i + 1]),
                                                        (route2_unique[j], route2_unique[j + 1]))

                dist_factor = \
                    (dist_to_stadium(route1_unique[i]) + dist_to_stadium(route1_unique[i + 1])) / 2 + \
                    (dist_to_stadium(route2_unique[j]) + dist_to_stadium(route2_unique[j + 1])) / 2 \
                    # ** (1 / 3)

                if segment_dist_sum_cur < threshold_dist_miles:
                    total_sim_segment_count += 1.0 * dist_factor
                elif segment_dist_sum_cur < threshold_dist_miles * 2:
                    total_sim_segment_count += 0.4 * dist_factor
                elif segment_dist_sum_cur < threshold_dist_miles * 3:
                    total_sim_segment_count += 0.2 * dist_factor

        cumul_point1 = utils_map.cumul_point(route1_unique)
        cumul_point2 = utils_map.cumul_point(route2_unique)
        angle = utils_map.angle_diff_points(route1_unique[0], cumul_point1, cumul_point2)
        return total_sim_segment_count * (1 - angle / math.pi)
    else:
        raise Exception('Unknown metric')


number_of_paths = len(coord_list)
# number_of_paths = 100
distance_matrix = np.zeros((number_of_paths, number_of_paths))
for i in range(number_of_paths):
    for j in range(i, number_of_paths):
        if i == j:
            distance = 0.0
        else:
            distance = compute_distance(coord_list[i], coord_list[j], 'sim_segments')
        distance_matrix[i][j] = distance
        distance_matrix[j][i] = distance
#         print('Elem done')
    print('Row done #' + str(i))

with open('data_routes_pickle/sim_matrix_sim_segments_1_mod23_cumul_1k', 'wb') as f:
    pickle.dump(distance_matrix, f)
