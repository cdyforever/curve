import numpy as np
import math

STROKE_SAMPLE_NUM = {'撇': [0.25], '点': [0.25], '横': [1], '捺': [0.25], '竖': [1], 
                 '提画': [0.5, 0.25], '卧钩': [0.25, 0.25], '反捺': [0.25], '右点': [0.25], 
                 '垂露竖': [1], '左点': [0.1], '平捺': [0.25], '平撇': [0.25], '弯钩': [0.25, 0.1], 
                 '悬针竖': [1], '提': [0.5], '撇折': [0.25, 0.25], 
                 '撇点': [0.25, 0.1], '斜捺': [0.25], '斜撇': [0.25], '斜钩': [0.25, 0.1], 
                 '横折': [0.5, 0.25], '横折弯': [0.5, 0.25, 0.25], '横折弯钩': [0.5, 0.25, 0.25, 0.1], 
                 '横折折': [0.5, 0.25, 0.25], '横折折折': [0.5, 0.25, 0.25, 0.25], 
                 '横折折折钩': [0.5, 0.25, 0.25, 0.25, 0.1], '横折折撇': [0.5, 0.25, 0.25, 0.25], 
                 '横折提': [0.5, 0.25, 0.25], '横折钩': [0.5, 0.25, 0.1], 
                 '横撇': [0.5, 0.25], '横撇弯钩': [0.5, 0.25, 0.25, 0.1], '横斜钩': [0.5, 0.25, 0.1], 
                 '横钩': [0.5, 0.1], '短撇': [0.25], '短横': [1], '短竖': [1], 
                 '竖弯': [0.5, 0.25], '竖弯钩': [0.8, 0.25, 0.1], '竖折': [0.5, 0.5], 
                 '竖折折': [0.5, 0.25, 0.25], '竖折折钩': [0.5, 0.25, 0.25, 0.1], 
                 '竖折撇': [0.5, 0.25, 0.25], '竖提': [0.5, 0.2], 
                 '竖撇': [0.25], '竖钩': [0.8, 0.1], '长横': [1]}


def triple_label(value, thresh):
    if value > thresh:
        return 'more'
    elif value < -thresh:
        return 'less'
    else:
        return None

def norm_curvature(curvature, t=0.015):
    curvature_filtered = curvature - np.mean(curvature)
    curvature_filtered[np.where(np.abs(curvature_filtered) < t)] = 0
    return curvature_filtered

def compute_bend(user_curv, std_curv, frag_length, seg_multi = 1.0,
                 curv_fit_t: float = 0.2, curv_status_t: float = 0.01):
    """ 判断弯曲情况
        弯曲度一致性：user与std的弯曲度按位相减，对每一位取绝对值后求平均，若结果小于阈值则一致；反之相反
        弯曲度方向一致性：若user与std的弯曲度按位相加后的和大于等于相减后的和，则方向一致；反之相反
        弯曲度状态：user与std的弯曲度取标准差后相减，若高于小于阈值表示过弯；小于阈值表示过直；绝对值小于阈值表示正常
        输入：
            user_curv: user弯曲度
            std_curv: std弯曲度
            curv_fit_t: 弯曲度一致性阈值
            curv_status_t: 弯曲状态阈值
        返回：
            delta_curv: 弯曲度差异
            curv_fit_label: 弯曲度一致性标签(True/False)
            curv_direct_label: 弯曲度方向一致性标签(True/False)
            curv_status_label: 弯曲度状态:过于弯曲、过于笔直、锯齿('more'/'less'/None)
    """
    
    
    # print("user curv ", user_curv)
    user_curv = norm_curvature(user_curv)
    # print("frag_length : ", frag_length)
    std_curv = norm_curvature(std_curv)
    diff_std = np.std(user_curv) - np.std(std_curv)
    # print("diff result : ", user_curv - std_curv)
    # print("user curv ", user_curv)
    # print("user std ", np.std(user_curv))
    delta_curv_v1 = np.sum(np.abs(user_curv - std_curv))
    # print("user std delta is : ", delta_curv_v1)
    delta_curv_v2 = np.sum(np.abs(user_curv))
    # print("user horizon delta is : ", delta_curv_v2)
    curv_result = min(delta_curv_v1, delta_curv_v2)
    # curv_result = delta_curv_v1
    curv_result = curv_result * math.pow(frag_length, 0.95)
    curv_result = curv_result * seg_multi
    # print("final delta is : ", curv_result)
    # accum_curv = np.sum(np.abs(user_curv + std_curv))
    curv_fit_label = True if curv_result < curv_fit_t else False
    # curv_status_label = triple_label(diff_std, curv_status_t)
    # if curv_fit_label is True:
    curv_label = None
    return curv_result, curv_label
