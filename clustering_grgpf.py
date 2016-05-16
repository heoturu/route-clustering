# coding=utf-8
import bisect

import pickle
import random

import math
import numpy as np

import utils_map
from utils_dist import *

class GRGPF:
    compute_distance = None

    def __init__(self, distance_measure, get_point_from_docid,
                 limit_subnodes, limit_clusters_per_leaf, limit_total_clusters, limit_total_nodes, k,
                 sample_size, get_next_threshold):
        self.limit_subnodes = limit_subnodes
        self.limit_clusters_per_leaf = limit_clusters_per_leaf
        self.limit_total_clusters = limit_total_clusters
        self.limit_total_nodes = limit_total_nodes
        self.distance_measure = distance_measure
        self.threshold = get_next_threshold()
        self.get_next_threshold = get_next_threshold
        self.sample_size = sample_size
        self.get_point_from_docid = get_point_from_docid
        self.k = k
        self.root = None

    def recompute(self, only_samples):
        self.root.recompute(only_samples)

    def create_stdtree(self):
        def dfs(node):
            if isinstance(node, GRGPF.InteriorNode):
                r = []
                for s in node.subnodes:
                    r.append(dfs(s))
                return r
            elif isinstance(node, GRGPF.Leaf):
                r = []
                for c in node.clusters:
                    r.append(dfs(c))
                return r
            elif isinstance(node, GRGPF.Cluster):
                return node.points
            else:
                assert False

        return dfs(self.root)

    def create_cluster_list(self):
        def mydfs(node):
            if isinstance(node, GRGPF.InteriorNode):
                for i in node.subnodes:
                    mydfs(i)
            elif isinstance(node, GRGPF.Leaf):
                for i in node.clusters:
                    mydfs(i)
            elif isinstance(node, GRGPF.Cluster):
                mydfs.result.append(node.points)
            else:
                assert False

        mydfs.result = []
        mydfs(self.root)
        return mydfs.result

    def add_point(self, p):
        if self.root is None:
            cluster = self.Cluster.init_base([p], self.k, self.distance_measure, (lambda: self.threshold), self.get_point_from_docid)

            # k, compute_distance, get_threshold, get_point_from_docId
            self.root = self.Leaf([cluster], self.distance_measure, lambda: self.limit_clusters_per_leaf)
        else:
            retval = self.root.add_point(p)
            if retval == 2:
                self.root = self.InteriorNode(self.root.split(), self.distance_measure, (lambda: self.limit_subnodes), self.sample_size)

            if self.root.get_cluster_count() > self.limit_total_clusters or \
               self.root.get_node_count() > self.limit_total_nodes:
                self.threshold = self.get_next_threshold()
                print("Start merge with new threshold " + str(self.threshold))
                print("Before: " + str(self.root.get_cluster_count()) + " " + str(self.root.get_node_count()))

                clusters = self.root.get_clusters()
                new_root = self.Leaf([clusters[0]], self.distance_measure, lambda: self.limit_clusters_per_leaf)

                for idx, cluster in enumerate(clusters):
                    if idx == 0:
                        continue
                    retval = new_root.add_cluster(cluster)
                    if retval is True:
                        new_root = self.InteriorNode(new_root.split(), self.distance_measure, (lambda: self.limit_subnodes), self.sample_size)
                self.root = new_root

                print("After: " + str(self.root.get_cluster_count()) + " " + str(self.root.get_node_count()))

    class Cluster:
        def __init__(self):
            self.N = 0
            self.k = 0
            self.compute_distance = lambda x, y: 0
            self.get_threshold = lambda x: 0
            self.get_point_from_docId = lambda x: None
            self.clustroid = ()
            self.clustroid_rowsum = 0
            self.nearest = []
            self.nearest_rowsums = []
            self.furthest = []
            self.furthest_rowsums = []
            self.points = []
            self.point_storage = None

        def _verify_struct(self):
            return # Debug purposes
            assert len(self.points) == self.N
            assert self.clustroid is not None
            if self.N >= 2 * self.k + 1:
                assert len(self.nearest) == self.k
                assert len(self.nearest_rowsums) == self.k
                assert len(self.furthest) == self.k
                assert len(self.furthest_rowsums) == self.k
                assert self.point_storage is None
            else:
                assert self.point_storage is not None

        @classmethod
        def init_merge(cls, cluster1, cluster2):
            if cluster1.point_storage is None:
                if cluster2.point_storage is None:
                    return cls._init_merge_plain(cluster1, cluster2)
                else:
                    return cls._init_merge_copy(cluster1, cluster2.point_storage)
            else:
                if cluster2.point_storage is None:
                    return cls._init_merge_copy(cluster2, cluster1.point_storage)
                else:
                    return cls.init_base(cluster1.point_storage + cluster2.point_storage, cluster1.k,
                                         cluster1.compute_distance,
                                         cluster1.get_threshold, cluster1.get_point_from_docId)

        @classmethod
        def _init_merge_copy(cls, cluster, new_points):
            self = cls()
            self.N = cluster.N
            self.k = cluster.k
            self.compute_distance = cluster.compute_distance
            self.get_threshold = cluster.get_threshold
            self.get_point_from_docId = cluster.get_point_from_docId

            self.points = list(cluster.points) # copy
            self.nearest = list(cluster.nearest) # copy
            self.furthest = list(cluster.furthest) # copy
            self.nearest_rowsums = list(cluster.nearest_rowsums) # copy
            self.furthest_rowsums = list(cluster.furthest_rowsums) # copy
            if cluster.point_storage is not None:
                self.point_storage = list(cluster.point_storage)
            else:
                self.point_storage = None
            self.clustroid = cluster.clustroid
            self.clustroid_rowsum = cluster.clustroid_rowsum

            for p in new_points:
                self.add_point(p)

            self._verify_struct()
            return self

        @classmethod
        def _init_merge_plain(cls, cluster1, cluster2):
            self = cls()
            self.N = cluster1.N + cluster2.N
            self.k = cluster1.k
            self.compute_distance = cluster1.compute_distance
            self.get_threshold = cluster1.get_threshold
            self.get_point_from_docId = cluster1.get_point_from_docId
            self.points = cluster1.points + cluster2.points

            # Select new point
            all_points_1 = [(cluster1.clustroid, cluster1.clustroid_rowsum)] + \
                           list(zip(cluster1.nearest, cluster1.nearest_rowsums)) + \
                           list(zip(cluster1.furthest, cluster1.furthest_rowsums))

            all_points_2 = [(cluster2.clustroid, cluster2.clustroid_rowsum)] + \
                           list(zip(cluster2.nearest, cluster2.nearest_rowsums)) + \
                           list(zip(cluster2.furthest, cluster2.furthest_rowsums))

            # Update rowsums
            distance_c1_c2 = self.compute_distance(cluster1.clustroid, cluster2.clustroid) ** 2
            for idx in range(0, len(all_points_1)):
                all_points_1[idx] = (all_points_1[idx][0],
                                     all_points_1[idx][1] +
                                     cluster1.N * (self.compute_distance(cluster2.clustroid,
                                                                         all_points_1[idx][0]) ** 2 + distance_c1_c2) +
                                     cluster2.clustroid_rowsum)
            for idx in range(0, len(all_points_2)):
                all_points_2[idx] = (all_points_2[idx][0],
                                     all_points_2[idx][1] +
                                     cluster2.N * (self.compute_distance(cluster1.clustroid,
                                                                         all_points_2[idx][0]) ** 2 + distance_c1_c2) +
                                     cluster1.clustroid_rowsum)

            all_points = all_points_1 + all_points_2
            min_point = None
            min_rowsum = None
            for point, rowsum in all_points:
                if min_rowsum is None or min_rowsum > rowsum:
                    min_point = point
                    min_rowsum = rowsum

            # We found the rowsum
            self.clustroid = min_point
            self.clustroid_rowsum = min_rowsum

            all_points = [(x[0], x[1], self.compute_distance(self.clustroid, x[0])) for x in all_points if
                          x[0] != self.clustroid]
            all_points = sorted(all_points, key=lambda x: x[2])

            self.nearest = [x[0] for x in all_points[0:self.k]]
            self.nearest_rowsums = [x[1] for x in all_points[0:self.k]]

            self.furthest = list(reversed([x[0] for x in all_points[len(all_points) - self.k:len(all_points)]]))
            self.furthest_rowsums = list(reversed([x[1] for x in all_points[len(all_points) - self.k:len(all_points)]]))

            self._verify_struct()

            return self

        @classmethod
        def init_base(cls, initial_points, k, compute_distance, get_threshold, get_point_from_docId):
            self = cls()
            self.N = len(initial_points)
            self.k = k
            self.compute_distance = compute_distance
            self.get_threshold = get_threshold
            self.get_point_from_docId = get_point_from_docId

            self.clustroid = None
            self.clustroid_rowsum = None

            self.nearest = []
            self.nearest_rowsums = []

            self.furthest = []
            self.furthest_rowsums = []

            self.points = []

            self._recompute(initial_points)
            self._verify_struct()

            return self

        def recompute(self):
            self._recompute([self.get_point_from_docId(x) for x in self.points])
            self._verify_struct()

        def _recompute(self, points):
            """
            Recompute completely the representation of the clusters from all the points in it
            :param points:
            :return:
            """
            dists = np.zeros((len(points), len(points)))
            for i, p1 in enumerate(points):
                dists[i][i] = 0.0
                for j, p2 in enumerate(points[i + 1:]):
                    dists[i][i + 1 + j] = self.compute_distance(p1, p2) ** 2
                    dists[i + 1 + j][i] = dists[i][i + 1 + j]
            rowsums = [sum(i) for i in dists]

            self.clustroid = None
            self.clustroid_rowsum = None
            for i, rowsum in enumerate(rowsums):
                if self.clustroid_rowsum is None or rowsum < self.clustroid_rowsum:
                    self.clustroid = i
                    self.clustroid_rowsum = rowsum

            if len(points) >= 2 * self.k + 1:
                order = sorted(range(0, len(points)), key=lambda x: dists[self.clustroid][i])
                self.nearest = order[1:self.k + 1]
                self.nearest_rowsums = [rowsums[i] for i in self.nearest]

                self.furthest = [order[x] for x in reversed(order[len(points) - self.k:len(points)])]
                self.furthest_rowsums = [rowsums[i] for i in self.furthest]

                self.clustroid = points[self.clustroid]
                self.nearest = [points[x] for x in self.nearest]
                self.furthest = [points[x] for x in self.furthest]
                self.point_storage = None
            else:
                self.clustroid = points[self.clustroid]
                self.point_storage = points
            self.points = [x[0] for x in points]
            self._verify_struct()

        def get_radius(self):
            return math.sqrt(self.clustroid_rowsum / float(self.N))

        def add_point(self, new_point):
            """
            :param new_point:
            :return: False if there is no need to split, True else
            """
            self._verify_struct()
            if len(self.points) >= 2 * self.k + 1:
                distance_to_clustroid = self.compute_distance(new_point, self.clustroid) ** 2
                new_point_rowsum = self.clustroid_rowsum + self.N * distance_to_clustroid

                self.N += 1
                self.points.append(new_point[0])
                for i in range(0, self.k):
                    self.furthest_rowsums[i] += self.compute_distance(new_point, self.furthest[i]) ** 2
                    self.nearest_rowsums[i] += self.compute_distance(new_point, self.nearest[i]) ** 2
                self.clustroid_rowsum += distance_to_clustroid

                # Update nearest or furthest if needed
                distance_nearest = [(self.compute_distance(self.nearest[i], self.clustroid) ** 2, i) for i in
                                    range(0, self.k)] # should be sorted by
                # Invariant
                distance_furthest = [(self.compute_distance(self.furthest[i], self.clustroid) ** 2, i) for i in
                                     range(0, self.k)] # should be sorted by
                # Invariant
                bisect.insort_left(distance_nearest, (distance_to_clustroid, -1))
                bisect.insort_left(distance_furthest, (distance_to_clustroid, -1))

                self.nearest = [(self.nearest[i] if i != -1 else new_point) for _, i in distance_nearest]
                self.nearest_rowsums = [(self.nearest_rowsums[i] if i != -1 else new_point_rowsum) for _, i in
                                        distance_nearest]

                self.furthest = [(self.furthest[i] if i != -1 else new_point) for _, i in distance_furthest]
                self.furthest_rowsums = [(self.furthest_rowsums[i] if i != -1 else new_point_rowsum) for _, i in
                                         distance_furthest]

                # Verify if self.clustroid is still the clustroid
                min_p = -1
                min_rowsum = self.clustroid_rowsum
                for i in range(0, len(self.nearest)):
                    if min_rowsum > self.nearest_rowsums[i]:
                        min_rowsum = self.nearest_rowsums[i]
                        min_p = i

                if min_p != -1:
                    new_clustroid = self.nearest[min_p]
                    new_clustroid_rowsum = self.nearest_rowsums[min_p]

                    all_distinct_points = {x for x in
                                           [i[0] for i in self.nearest] +
                                           [i[0] for i in self.furthest] +
                                           [new_point[0], self.clustroid[0]]}
                    distinct_points = []
                    for pt, rowsum in \
                            [(self.clustroid, self.clustroid_rowsum), (new_point, new_point_rowsum)] + \
                            list(zip(self.nearest, self.nearest_rowsums)) + \
                            list(zip(self.furthest, self.furthest_rowsums)):
                        if pt[0] != new_clustroid[0] and pt[0] in all_distinct_points:
                            all_distinct_points.remove(pt[0])
                            distinct_points.append((pt, rowsum, self.compute_distance(new_clustroid, pt) ** 2))
                    distinct_points = sorted(distinct_points, key=lambda x: x[2])

                    self.clustroid = new_clustroid
                    self.clustroid_rowsum = new_clustroid_rowsum

                    self.nearest = [x[0] for x in distinct_points[0:self.k]]
                    self.nearest_rowsums = [x[1] for x in distinct_points[0:self.k]]

                    # TODO Optimize
                    self.furthest = \
                        [x[0] for x in reversed(distinct_points[len(distinct_points) - self.k:len(distinct_points)])]
                    self.furthest_rowsums = \
                        [x[1] for x in reversed(distinct_points[len(distinct_points) - self.k:len(distinct_points)])]
                else:
                    # Ensure we still have k nearest/furthest points
                    self.nearest = self.nearest[0:self.k]
                    self.nearest_rowsums = self.nearest_rowsums[0:self.k]
                    self.furthest = self.furthest[1:self.k + 1]
                    self.furthest_rowsums = self.furthest_rowsums[1:self.k + 1]
            else:
                self.N += 1
                self._recompute(self.point_storage + [new_point])

            # Done!
            self._verify_struct()
            return self.get_radius() > self.get_threshold()

        def split(self):
            p1 = None
            p2 = None
            dist = None
            if self.point_storage is not None:
                for i in range(0, len(self.point_storage)):
                    for j in range(i + 1, len(self.point_storage)):
                        d = self.compute_distance(self.point_storage[i], self.point_storage[j])
                        if dist is None or dist > d:
                            p1 = self.point_storage[i]
                            p2 = self.point_storage[j]
                            dist = d

            else:
                for i in range(0, self.k):
                    for j in range(i + 1, self.k):
                        d = self.compute_distance(self.furthest[i], self.furthest[j])
                        if dist is None or dist > d:
                            p1 = self.furthest[i]
                            p2 = self.furthest[j]
                            dist = d

            points1 = [p1]
            points2 = [p2]
            for i, x in enumerate(self.points):
                if x == p1[0] or x == p2[0]:
                    continue

                p = self.get_point_from_docId(x) if self.point_storage is None else self.point_storage[i]
                d1 = self.compute_distance(p1, p)
                d2 = self.compute_distance(p2, p)

                if d1 < d2:
                    points1.append(p)
                else:
                    points2.append(p)

            return [self.__class__.init_base(points1, self.k, self.compute_distance, self.get_threshold, self.get_point_from_docId),
                    self.__class__.init_base(points2, self.k, self.compute_distance, self.get_threshold, self.get_point_from_docId)]

    class Leaf:
        def __init__(self, clusters, compute_distance, get_max_subnodes):
            self.clusters = clusters
            self.compute_distance = compute_distance
            self.get_max_subnodes = get_max_subnodes

        def add_point(self, p):
            """
            :param p:
            :return: 0 if no need to do anything, 1 if should re-sample, 2 if need to split
            """
            nearest = None
            dist = None
            for c in self.clusters:
                d = self.compute_distance(c.clustroid, p)
                if dist is None or dist > d:
                    dist = d
                    nearest = c

            if nearest.add_point(p): # should split
                toadd = nearest.split()
                self.clusters += toadd
                self.clusters.remove(nearest)

                if len(self.clusters) > self.get_max_subnodes():
                    return 2
                else:
                    return 1
            return 0

        def get_cluster_count(self):
            return len(self.clusters)

        def get_node_count(self):
            return 1

        def get_clusters(self):
            return self.clusters

        def add_cluster(self, new_cluster):
            nearest = None
            dist = None
            for c in self.clusters:
                d = self.compute_distance(c.clustroid, new_cluster.clustroid)
                if dist is None or dist > d:
                    dist = d
                    nearest = c

            # Try a merge
            merged = nearest.__class__.init_merge(nearest, new_cluster)
            if merged.get_radius() > merged.get_threshold():
                self.clusters.append(new_cluster)
            else:
                self.clusters.remove(nearest)
                self.clusters.append(merged)

            return len(self.clusters) > self.get_max_subnodes()

        def recompute(self, only_samples):
            if only_samples:
                return

            for c in self.clusters:
                c.recompute()

        def get_size(self):
            return len(self.clusters)

        def get_random_sample(self, size):
            random.shuffle(self.clusters)
            return [x.clustroid for x in self.clusters[0:size]]

        def split(self):
            p1 = None
            p2 = None
            dist = None
            for i in range(0, len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    d = self.compute_distance(self.clusters[i].clustroid, self.clusters[j].clustroid)
                    if dist is None or dist > d:
                        dist = d
                        p1 = self.clusters[i]
                        p2 = self.clusters[j]

            points_1 = [p1]
            points_2 = [p2]
            for c in self.clusters:
                if c == p1 or c == p2:
                    continue
                d1 = self.compute_distance(p1.clustroid, c.clustroid)
                d2 = self.compute_distance(p2.clustroid, c.clustroid)
                if d1 < d2:
                    points_1.append(c)
                else:
                    points_2.append(c)

            return [self.__class__(points_1, self.compute_distance, self.get_max_subnodes),
                    self.__class__(points_2, self.compute_distance, self.get_max_subnodes)]

        def get_clustroid_sample(self):
            return [x.clustroid for x in self.clusters]

    class InteriorNode:
        def __init__(self, subnodes, compute_distance, get_max_subnodes, sample_size):
            self.subnodes = subnodes
            self.compute_distance = compute_distance
            self.get_max_subnodes = get_max_subnodes
            self.sample_size = sample_size
            self.sample = []
            self._produce_sample()

        def _produce_sample(self, size=None):
            if size is None:
                size = self.sample_size
            self.sample = []
            total = sum([s.get_size() for s in self.subnodes])
            for s in self.subnodes:
                count = max(1, int((s.get_size() * size) / total))
                self.sample += [(c, s) for c in s.get_random_sample(count)]

        def get_cluster_count(self):
            return sum([x.get_cluster_count() for x in self.subnodes])

        def get_node_count(self):
            return 1 + sum([x.get_node_count() for x in self.subnodes])

        def get_clusters(self):
            data = []
            for x in self.subnodes:
                data += x.get_clusters()
            return data

        def get_size(self):
            return len(self.sample)

        def get_random_sample(self, size):
            random.shuffle(self.sample)
            return [x[0] for x in self.sample[0:size]]

        def add_point(self, p):
            """
            :param p:
            :return: 0 if there is no need to do anything, 1 if need to resample, 2 if need to split
            """
            nearest = None
            dist = None
            for clustroid, node in self.sample:
                d = self.compute_distance(p, clustroid)
                if dist is None or dist > d:
                    dist = d
                    nearest = node
            retval = nearest.add_point(p)
            if retval == 1: # should resample
                self._produce_sample()
                return 1
            elif retval == 2: # should split
                toadd = nearest.split()
                self.subnodes += toadd
                self.subnodes.remove(nearest)
                self._produce_sample()

                if len(self.subnodes) > self.get_max_subnodes():
                    return 2
                return 1
            return 0

        def add_cluster(self, c):
            nearest = None
            dist = None
            for clustroid, node in self.sample:
                d = self.compute_distance(c.clustroid, clustroid)
                if dist is None or dist > d:
                    dist = d
                    nearest = node
            retval = nearest.add_cluster(c)
            if not retval:
                self._produce_sample()
                return False
            else:
                toadd = nearest.split()
                self.subnodes += toadd
                self.subnodes.remove(nearest)
                self._produce_sample()

                if len(self.subnodes) > self.get_max_subnodes():
                    return True
                return False

        def recompute(self, only_samples):
            for s in self.subnodes:
                s.recompute(only_samples)
            self._produce_sample()

        def split(self):
            self._produce_sample(1) # force to have only one sample for each subnode

            p1 = None
            p2 = None
            dist = None
            for i in range(0, len(self.sample)):
                for j in range(i + 1, len(self.sample)):
                    d = self.compute_distance(self.sample[i][0], self.sample[j][0])
                    if dist is None or dist > d:
                        dist = d
                        p1 = self.sample[i]
                        p2 = self.sample[j]

            points_1 = [p1[1]]
            points_2 = [p2[1]]
            for c, n in self.sample:
                if c == p1[0] or c == p2[0]:
                    continue
                d1 = self.compute_distance(p1[0], c)
                d2 = self.compute_distance(p2[0], c)
                if d1 < d2:
                    points_1.append(n)
                else:
                    points_2.append(n)

            return [self.__class__(points_1, self.compute_distance, self.get_max_subnodes, self.sample_size),
                    self.__class__(points_2, self.compute_distance, self.get_max_subnodes, self.sample_size)]


def compute_distance(route1, route2):
    route1_stopind = len(route1)
    for i, coord in enumerate(route1):
        if coord == -1:
            route1_stopind = i
            break
    route1_unique = route1[1:route1_stopind]
    route1_unique = list(zip(route1_unique[::2], route1_unique[1::2]))

    route2_stopind = len(route2)
    for i, coord in enumerate(route2):
        if coord == -1:
            route2_stopind = i
            break
    route2_unique = route2[1:route2_stopind]
    route2_unique = list(zip(route2_unique[::2], route2_unique[1::2]))

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
    return 135 - total_sim_segment_count * (1 - angle / math.pi)


def threshold_cosine(initial, max, max_count):
    def t():
        t.count += 1
        return float(initial) + float((max - initial) * min(t.count, max_count)) / float(max_count)

    t.count = 0
    return t


with open('data_routes_pickle/routes_coord_any_3_contains_nocycles_angle_1.2_1k', 'rb') as f:
    routes_coord = pickle.load(f)

routes_coord = [[num for coords in route for num in coords] for route in routes_coord]

for i, route in enumerate(routes_coord):
    route.insert(0, i)

max_route_len = max([len(route) for route in routes_coord])
for i, route in enumerate(routes_coord):
    rng = range(max_route_len - len(route))
    for _ in rng:
        route.append(-1)


def route_by_id(route_id):
    return routes_coord[route_id]


if __name__ == "__main__":
    grgpf = GRGPF(compute_distance, route_by_id,
                  limit_subnodes=10, limit_clusters_per_leaf=2, limit_total_clusters=9, limit_total_nodes=200, k=15,
                  sample_size=30, get_next_threshold=threshold_cosine(116, 124, 10))

    for i, route in enumerate(routes_coord):
        print(i)

        if i != 0 and i % 30 == 0: # recalcul des samples tout les 100 points
            print("Recomputing samples")
            grgpf.recompute(True)
            print("Recomputing done")

        if i != 0 and i % 50 == 0: # recalcul complet des representations tout les 5000 points
            print("Recomputing representations")
            grgpf.recompute(False)
            print("Recomputing done")

        if i > 100: # limite sur le nombre de document qu'on ajoute
            break

        grgpf.add_point(route)

    print(*(grgpf.create_cluster_list()), sep='\n')
    # print(grgpf.create_stdtree())
    print("smth")
