import math

PEAK_INCLINE_FACTOR = 0.375
PEAK_FLATTEN_FACTOR = 1.25

SEGMENTS_STATIS = {
    'delta_curv': {'lambda': 0.6054, 'y_mean': -1.1639, 'y_std': 0.1161, 'y_max': -0.7828, 'y_min': -1.4266}
}

def delta_to_nd_grade(x, statis=SEGMENTS_STATIS["delta_curv"], p_incline=PEAK_INCLINE_FACTOR, p_flatten=PEAK_FLATTEN_FACTOR):
    """ 差异转化为分数
        输入：
            x: 待转换差异数值
            statis: 用于分数转换的参数集，keys包括：lambda、y_mean、y_std、y_max、y_min
            p_incline: 峰度偏移系数，用于调整分数分布
            p_flatten: 峰度扁平系数，用于调整分数分布
        输出：
            score: 0~1区间浮点数
    """
    lmbda = statis['lambda']
    y_mean, y_std = statis['y_mean'], statis['y_std']
    y_max, y_min = statis['y_max'], statis['y_min']
    # boxcox公式
    y = (abs(x)**lmbda - 1) / lmbda if lmbda != 0 else math.log(x)
    y_s = (y - y_mean) / y_std  # 转换为标准正太分布
    # 按标准正态分布转换上下界
    y_s_max = (y_max - y_mean) / y_std
    y_s_min = (y_min - y_mean) / y_std
    # 归一化至0-1区间
    # TODO: 偏移中心
    grade = y_s / (y_s_max - y_s_min) * p_flatten + p_flatten * 0.5  # 转换为0-1数值，并反转为分数
    grade = 1 - grade  # 反转，使分数与差异成反比
    grade = min(max(grade, 0), 1)  # 消除由分布平坦和浮点数精度导致的越界
    grade = math.pow(grade, p_incline)
    return grade

def grade_func(delta_curv):
    grade = delta_to_nd_grade(delta_curv)
    return grade