from geopy.distance import vincenty

stadium_coords = 30.28725, 59.95271
threshold_dist_miles = 1.0


def dist_to_stadium(point):
    return vincenty(stadium_coords, point).miles


def dist_vinc(point1, point2):
    return vincenty(point1, point2).miles


def dist_vinc_pair(pair):
    return vincenty(pair[0], pair[1]).miles


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
