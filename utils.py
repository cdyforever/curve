import math
import numpy as np

def get_project_length(vec_a, vec_b):
    """ 获取向量A到向量B的投影长度 """
    project_len = np.dot(vec_a, vec_b) / np.dot(vec_b, vec_b)
    return project_len

def get_vector_radian(y, x):
    """ 向量弧度 """
    return (math.atan2(y, x) / math.pi) % 2

def get_focus(points_y, points_x):
    """ 计算图像中心点，可以是笔划，也可以是整字，或者部件 """
    focus = (np.mean(points_y), np.mean(points_x))
    return focus

def detach_pix_from_mat(mat, pix_thresh=63):
    h, w = mat.shape
    points_y, points_x = np.where(mat > pix_thresh)
    points_y = points_y / h
    points_x = points_x / w
    return points_y, points_x


def get_dot2line_dist(point, line_start, line_end):
    """ 获取点到直线距离 """
    v1 = line_start - point
    v2 = line_end - point
    distance = np.cross(v1, v2) / np.linalg.norm(line_start - line_end)
    return distance

def euclidean_distance(pt1, pt2):
    """ 两点的欧氏距离 """
    y1, x1 = pt1[0], pt1[1]
    y2, x2 = pt2[0], pt2[1]
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


if __name__ == "__main__":
    point = np.array([0, 2])
    start = np.array([-2, 0])
    end = np.array([3, 1])
    res = get_dot2line_dist(point, start, end)
