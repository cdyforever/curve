import math
import cv2
import numpy as np
import scipy.interpolate as interpolate
from scipy.optimize import leastsq



def leastsq_func(p, x):
    k, b = p
    return k * x + b

def leastsq_error(p, x, y):
    return leastsq_func(p, x) - y

def rotate_array(track_array):
    """
    将原始点集旋转到水平，这里假设原始点集的最佳拟合方式为直线
    """
    track_x_array = track_array[:, 0]
    track_y_array = track_array[:, 1]
    x_range = max(track_x_array) - min(track_x_array)
    y_range = max(track_y_array) - min(track_y_array)
    if y_range > x_range * 2:
        track_array = track_array[:, ::-1]
        track_x_array = track_array[:, 0]
        track_y_array = track_array[:, 1]
    param_start = [1.0, 1.0]
    Para = leastsq(leastsq_error, param_start, args=(track_x_array, track_y_array))
    k, b = Para[0]
    angle = - math.atan(k) * 180 / math.pi
    rotate_matrix = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=1.0)[:, :2]
    horizon_track_array = np.matmul(track_array, rotate_matrix)
    track_sort = np.argsort(horizon_track_array[:, 0])
    horizon_track_array = horizon_track_array[track_sort]
    return horizon_track_array

def filter_nearest(horizon_stroke_array):
    """
    近邻点过滤，通过采样拟合的前提是，笔划内的点间距大于最小间距 (默认256*256的图像 需要大于至少1一个像素)
    """
    if len(horizon_stroke_array) < 2:
        return horizon_stroke_array
    last_pts = horizon_stroke_array[0]
    for i in range(1, len(horizon_stroke_array)):
        if abs(horizon_stroke_array[i][0] - last_pts[0]) < 0.003:
            horizon_stroke_array[i] = 0
        else:
            last_pts = horizon_stroke_array[i]
    horizon_stroke_array = horizon_stroke_array[horizon_stroke_array[:, 0] != 0]
    return horizon_stroke_array

def track_sample_hist(track_array, sample_num):
    """
    对点集进行曲线拟合和等间隔采样，获取采样后的直方图 (y_hist)
    """
    # track_array = filter_nearest(track_array)
    _, inds = np.unique(track_array[:, 0], return_index=True)
    track_array = track_array[inds]
    # 长度不足4时，如下处理
    if len(track_array) < 4:
        return np.zeros((1, sample_num), dtype=np.float32)
    track_x_list = track_array[:, 0].tolist()
    track_y_list = track_array[:, 1].tolist()
    track_x_start, track_x_end = track_x_list[0], track_x_list[-1]
    track_y_start, track_y_end = track_y_list[0], track_y_list[-1]
    # 修改采样函数的采样方法，在大量数据上测试，表征采样结果
    interp_func = interpolate.interp1d(track_x_list, track_y_list, kind='linear')
    sample_x_array = np.linspace(track_x_start, track_x_end, num=sample_num, endpoint=True)
    sample_x_list = sample_x_array.tolist()
    sample_y_base_list = np.linspace(track_y_start, track_y_end, num=sample_num, endpoint=True).tolist()
    sample_y_list = interp_func(sample_x_list)
    sample_hist = (sample_y_list - sample_y_base_list) / (track_x_end - track_x_start)
    print("sample_hist ", sample_hist)
    return sample_hist










# def horizontal_track(track_array, matrix):
#     """
#     将原始点集旋转至水平，过滤近邻点，并保证点集的方向一致
#     """
#     print("src track ", track_array)
#     track_length = len(track_array)
#     track_array_xyb = np.hstack([track_array, np.ones((track_length, 1))])
#     horizon_track_array = np.matmul(track_array_xyb, matrix)
#     track_sort = np.argsort(horizon_track_array[:, 0])
#     sort_list = track_sort.tolist()
#     start_idx = sort_list.index(0)
#     end_idx = sort_list.index(track_length - 1)
#     if start_idx > end_idx:
#         tmp_idx = end_idx
#         end_idx = start_idx
#         start_idx = tmp_idx
#     horizon_track_array = horizon_track_array[track_sort[start_idx: end_idx + 1]]
#     print("horizon track ", horizon_track_array)
#     return horizon_track_array
