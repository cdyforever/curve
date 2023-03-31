import math
import cv2
import numpy as np
import scipy.interpolate as interpolate
from scipy.optimize import leastsq

MAX_IGNORE_RATE = 0.03
MIN_PIX_HINGE = 0.003
MAX_VALUE_CONCLUDE = 5


def leastsq_func(p, x):
    k, b = p
    return k * x + b

def leastsq_error(p, x, y):
    return leastsq_func(p, x) - y

def track_leastsq_y(track_array):
    """
    对点集进行曲线拟合和等间隔采样，获取采样后的直方图 (y_hist)
    """
    track_x_array = track_array[:, 0]
    track_y_array = track_array[:, 1]
    param_start = [1.0, 1.0]
    Para = leastsq(leastsq_error, param_start, args=(track_x_array, track_y_array))
    leastsq_y = leastsq_func(Para[0], track_x_array)
    k, b = Para[0]
    angle = - math.atan(k) * 180 / math.pi
    rotate_matrix = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=1.0)[:, :2]
    horizon_track_array = np.matmul(track_array, rotate_matrix)
    print(horizon_track_array)
    return


if __name__ == "__main__":
    tracklist = [[0, 0], [1, 1], [2, 2]]
    track_array = np.array(tracklist, dtype=np.float32)
    track_leastsq_y(track_array)