import numpy as np
import math

SEG_MULTI_COEFF = {'撇': [0.25], '点': [0.1], '横': [1], '捺': [0.25], '竖': [1], 
                 '提画': [0.25, 0.25], '卧钩': [0.25, 0.25], '反捺': [0.25], '右点': [0.1],
                 '垂露竖': [1], '左点': [0.1], '平捺': [0.25], '平撇': [0.25], '弯钩': [0.25, 0.1], 
                 '悬针竖': [1], '提': [0.5], '撇折': [0.25, 0.25], 
                 '撇点': [0.25, 0.1], '斜捺': [0.25], '斜撇': [0.25], '斜钩': [0.25, 0.1], 
                 '横折': [0.5, 0.25], '横折弯': [0.5, 0.25, 0.25], '横折弯钩': [0.5, 0.25, 0.25, 0.1], 
                 '横折折': [0.5, 0.25, 0.25], '横折折折': [0.5, 0.25, 0.25, 0.25], 
                 '横折折折钩': [0.5, 0.25, 0.25, 0.25, 0.1], '横折折撇': [0.5, 0.25, 0.25, 0.25], 
                 '横折提': [0.5, 0.25, 0.25], '横折钩': [0.5, 0.25, 0.1], 
                 '横撇': [0.5, 0.25], '横撇弯钩': [0.5, 0.25, 0.25, 0.1], '横斜钩': [0.5, 0.25, 0.1], 
                 '横钩': [0.5, 0.1], '短撇': [0.25], '短横': [0.8], '短竖': [0.8],
                 '竖弯': [0.5, 0.25], '竖弯钩': [0.5, 0.25, 0.1], '竖折': [0.5, 0.5], 
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


def norm_curvature(curvature, t=0.01):
    mask = (np.abs(curvature) - t) < 0
    curvature_filtered = curvature.copy()
    curvature_filtered[mask] = 0
    curvature_filtered -= np.mean(curvature_filtered)
    return curvature_filtered


def compute_bend(user_curv, std_curv, frag_length, seg_multi = 1.0,
                 curv_fit_t: float = 0.2, curv_status_t: float = 0.01):
    user_curv = norm_curvature(user_curv)
    std_curv = norm_curvature(std_curv)
    diff_std = np.std(user_curv) - np.std(std_curv)
    delta_curv_diff = np.sum(np.abs(user_curv - std_curv))
    delta_curv_undiff = np.sum(np.abs(user_curv + std_curv))
    delta_curv_self = np.sum(np.abs(user_curv))
    curv_result = min(delta_curv_diff, delta_curv_undiff, delta_curv_self)
    curv_result = curv_result * math.pow(frag_length, 0.9)
    curv_result = curv_result * seg_multi
    curv_fit_label = True if np.mean(curv_result) < curv_fit_t else False
    # curv_direct_label = True if accum_curv >= delta_curv else False
    curv_status_label = triple_label(diff_std, curv_status_t)

    if curv_fit_label is True:
        curv_label = None
    else:
        if curv_status_label is None:
            curv_label = 'shake'
        elif curv_status_label == 'more':
            curv_label = 'more'
        elif curv_status_label == 'less':
            curv_label = 'less'
        else:
            curv_label = None
    return curv_result, curv_label

