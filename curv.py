import numpy as np
import math
from collections import OrderedDict
import sys
sys.path.append('D:/Project/calligraphy-evaluation-algorithm')
sys.path.append('E:/Project/calligraphy-evaluation-algorithm')
from calligraphy_evaluation.functions.preprocess import WritingData
from calligraphy_evaluation.functions.preprocess.custom_type.feature import StrokeFeature
from utils import euclidean_distance, get_project_length, get_dot2line_dist, get_vector_radian, get_focus, detach_pix_from_mat
from calligraphy_evaluation.functions.algorithm.geom_feat.get_stroke_feat import CornerDetector
from calligraphy_evaluation.functions.algorithm.param import GEOM_PIX_T
from rotate import get_rotation_matrix, horizontal_track, track_sample_hist
import time
from bend import STROKE_SAMPLE_NUM

class GetStrokeCurvature(object):

    def __init__(self):
        self.bins = 12  # 采样数，一般不变
        self.length_multi_ratio = 0.5 # 长度关联系数
        self.min_len = 0.225  # 需要计算弧度的最小笔段长度，如果小于该长度，则不计算
        self.SIZE = 1024

    def __call__(self, wd: WritingData, corners, seq: list = None):
        """ 衡量笔划中每个（长）笔段的弯曲程度，用长度n的列表表示弯曲程度
            在笔段首尾连一条直线l，在l均匀取n-1个点，共组成n段；
            对每段末端做与l的垂线h，h与笔段轨迹的交点p到与l的交点p'的距离为列表每项的值
            输入：
                wd: 书写数据WritingData
                seq: 对于wd，按标准笔序排序后的列表
                corners: 对于wd，每个笔划的拐点在该笔划轨迹点序列中的索引
            返回：
                curvature: 弯曲度
        """
        curvature_in_strokes = OrderedDict()
        stroke_wds = wd.strokes
        if seq is None:
            seq = list(range(len(wd.strokes)))
        for idx in seq:
            curvature = self.get_curvature(stroke_wds[idx], corners[idx])
            curvature_in_strokes[idx] = curvature
        return curvature_in_strokes

    def get_curvature(self, stroke_wd, corner):
        track = stroke_wd.track
        curvature = []
        # 此处所有笔段统一考虑，对于较短笔段数值普遍较低
        # TODO: curvature_hist数值按笔划长度标准化
        for i in range(len(corner) - 1):
            seg_track = track[corner[i]: corner[i + 1] + 1]
            curvature.append(self._get_curvature_hist(seg_track))
        return curvature
    
    def _get_curvature_hist(self, track):
        if len(track) <= 1:
            return None
        seg_track_xy = track[:, ::-1]
        matrix = get_rotation_matrix(seg_track_xy)
        seg_horizon = horizontal_track(seg_track_xy, matrix)
        curv_hist = track_sample_hist(seg_horizon, 20)
        return curv_hist

    def _get_curvature_hist_old(self, track):
        if len(track) <= 1:
            return None
        # track = self._move_track(track=track)
        start, end = track[0], track[-1]
        vec_base = end - start
        len_base = euclidean_distance((0, 0), vec_base)
        key_pts = []
        fake_pts = []
        edge_pts = [start, end]
        # 对于首尾点距离较近的笔段，忽略弯曲度
        if len_base < self.min_len:
            key_dists = [0.] * self.bins
        else:
            len_bin = len_base / self.bins
            # 初始化
            mile_stone = len_bin  # 当前关键投影距离
            key_dists = []
            for pt1, pt2 in zip(track[:-1], track[1:]):
                vec = pt2 - pt1
                len_vec = get_project_length(vec, vec_base) * len_base
                len_cur = get_project_length(pt2 - start, vec_base) * len_base
                # 当前段长超过了关键位置，则截取至关键位置
                while len_cur >= mile_stone:
                    r = (len_vec - (len_cur - mile_stone)) / len_vec
                    brk_pt = pt1 + vec * r  # 按距离偏移比例找到准确轨迹点
                    key_pts.append(pt1)
                    fake_pts.append(brk_pt)
                    dist = get_dot2line_dist(brk_pt, start, end)
                    dist_norm = dist / len_base
                    key_dists.append(dist_norm)
                    mile_stone += len_bin
                    if len(key_dists) >= self.bins:  # 不考虑垂线位于vec_base延长线上的点
                        return np.array(key_dists), key_pts, fake_pts, edge_pts
            if len(key_dists) < self.bins:  # 添加由于浮点数精度导致的未加入的末尾点
                dist = get_dot2line_dist(end, start, end)
                dist_norm = dist / len_base
                key_dists.append(dist_norm)
                key_pts.append(end)
                fake_pts.append(end)
        return np.array(key_dists), key_pts, fake_pts, edge_pts



class GetStrokeFeature(object):

    def __init__(self, pix_thresh=GEOM_PIX_T):
        self.pix_thresh = pix_thresh
        self.corner_detect = CornerDetector()
        self.get_stroke_curvature = GetStrokeCurvature()

    @staticmethod
    def _get_lengths(dm, corners):
        seg_lens = []
        if len(corners) <= 1:
            return 0, []
        else:
            for i in range(len(corners) - 1):
                seg_lens.append(dm[corners[i]][corners[i + 1]])
            length = sum(seg_lens)
            length = length if length > 0 else 1e-7
            seg_len_ratios = [seg_len / length for seg_len in seg_lens]
            return length, seg_len_ratios
        
    @staticmethod
    def _get_radians(track, corners):
        """ 计算笔段的弧度 """
        radians = []
        for i in range(len(corners) - 1):
            pt1 = track[corners[i]]
            pt2 = track[corners[i + 1]]
            y, x = pt2[0] - pt1[0], pt2[1] - pt1[1]
            radian = get_vector_radian(y, x)
            radians.append(radian)
        return radians

    def __call__(self, cchar_wd, stro_seq):
        assert len(stro_seq) == len(cchar_wd.strokes)
        stroke_wds = cchar_wd.strokes
        results = []
        for stro_idx in stro_seq:
            stroke_wd = stroke_wds[stro_idx]
            track = stroke_wd.array[:, :2]
            corners = self.corner_detect.get_corners(stroke_wd)[0]
            length, seg_len_ratios = self._get_lengths(stroke_wd.dm, corners)
            # print("current length ratio is ", seg_len_ratios)
            points_y, points_x = detach_pix_from_mat(stroke_wd.mat, self.pix_thresh)
            curvature = self.get_stroke_curvature.get_curvature(stroke_wd, corners)
            result = {
                'stroke_idx': stro_idx,
                'radians': self._get_radians(track, corners),
                'corners': corners,
                'curvature': curvature,
                'len_ratio': seg_len_ratios,
                'length': length,
                'focus': get_focus(points_y, points_x),
            }
            stroke_feat = StrokeFeature(stroke_idx=result['stroke_idx'],
                                        stroke_len=result['length'],
                                        focus=result['focus'],
                                        radians=result['radians'],
                                        ratios=result['len_ratio'],
                                        corners=result['corners'],
                                        curvature=result['curvature'])
            results.append(stroke_feat)
        return results