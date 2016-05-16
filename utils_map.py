import numpy as np
from itertools import chain
import math
import fiona
from shapely.geometry import Point
from mpl_toolkits.basemap import Basemap


shp = fiona.open('data_maps_input/spb.shp')
bds = shp.bounds
shp.close()

ll = bds[0], bds[1]
ur = bds[2], bds[3]
coords = list(chain(ll, ur))
w, h = coords[2] - coords[0], coords[3] - coords[1]
zoom_out_frac = -0.3

m = Basemap(
    projection='tmerc',
    lon_0=30.5,
    lat_0=60.,
    ellps='WGS84',
    llcrnrlon=coords[0] + (coords[2] - coords[0]) * 0.06 - zoom_out_frac * w,
    llcrnrlat=coords[1] - zoom_out_frac * h,
    urcrnrlon=coords[2] + (coords[2] - coords[0]) * 0.06 + zoom_out_frac * w,
    urcrnrlat=coords[3] + zoom_out_frac * h,
    lat_ts=0,
    resolution='i',
    suppress_ticks=True)


def vector_diff(point1, point2):
    return [point1.x - point2.x, point1.y - point2.y]


def angle_diff_points(point0, point1, point2):
    map_point0, map_point1, map_point2 = \
        Point(m(point0[0], point0[1])), \
        Point(m(point1[0], point1[1])), \
        Point(m(point2[0], point2[1])),

    vec1 = vector_diff(map_point1, map_point0)
    vec2 = vector_diff(map_point2, map_point0)
    prod = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    if math.isclose(float(prod), 1, rel_tol=1e-9):
        return 0
    return math.acos(prod)


def angle_diff_max(route):
    angle_diff_max = 0
    for i in range(1, len(route) - 1):
        for j in range(i + 1, len(route)):
            angle_diff_cur = angle_diff_points(route[0], route[i], route[j])
            if angle_diff_cur > angle_diff_max:
                angle_diff_max = angle_diff_cur

    return angle_diff_max


def cumul_point(route):
    cumul_point = [0, 0]
    for i, point in enumerate(route[1:]):
        cumul_point[0] += point[0]
        cumul_point[1] += point[1]

    z = len(route) - 1
    return [cumul_point[0] / z, cumul_point[1] / z]