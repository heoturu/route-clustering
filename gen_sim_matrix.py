import numpy as np
import pickle
import math

from itertools import product
from itertools import groupby
from operator import itemgetter

from geopy.distance import vincenty

import utils


stadium_coords = 30.28725, 59.95271
def dist_to_stadium(point):
    return vincenty(stadium_coords, point).miles


def remove_dups_conseq(route):
    return list(map(itemgetter(0), groupby(route)))


threshold_dist_miles = 1.0

with open('data_routes_pickle/routes_coord_any_3_contains_nocycles_angle_1.2', 'rb') as f:
    routes_coord = pickle.load(f)

coord_list = routes_coord
# For 'norm' metric
# coord_list = [[num for coords in route for num in coords] for route in routes_coord]


def dist_vinc_pair(pair):
    return vincenty(pair[0], pair[1]).miles


def dist_vinc(point1, point2):
    return vincenty(point1, point2).miles


def similar_segments(segment1, segment2):
    d1 = dist_vinc(segment1[0], segment2[0])
    d2 = dist_vinc(segment1[1], segment2[1])

    d3 = dist_vinc(segment1[0], segment2[1])
    d4 = dist_vinc(segment1[1], segment2[0])

    if (d1 < threshold_dist_miles and d2 < threshold_dist_miles or
        d3 < threshold_dist_miles and d4 < threshold_dist_miles):
        return True

    return False


def segment_dist_sum(segment1, segment2):
    d1 = dist_vinc(segment1[0], segment2[0])
    d2 = dist_vinc(segment1[1], segment2[1])

    d3 = dist_vinc(segment1[0], segment2[1])
    d4 = dist_vinc(segment1[1], segment2[0])

    return min(max(d1, d2), max(d3, d4))


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

            most_sim_points = min(product(route1_set, route2_set), key=dist_vinc)

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

        cumul_point1 = utils.cumul_point(route1_unique)
        cumul_point2 = utils.cumul_point(route2_unique)
        angle = utils.angle_diff_points(route1_unique[0], cumul_point1, cumul_point2)
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

with open('data_routes_pickle/sim_matrix_sim_segments_1_mod23_cumul', 'wb') as f:
    pickle.dump(distance_matrix, f)
