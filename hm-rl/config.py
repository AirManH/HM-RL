import math

UNIT_LEN_KM = 10  # 单位公路的长度，单位是km
TIME_INTERVAL_DAY = 1  # 时间片
DAY_PER_YEAR = 365  # 一年的天数
MAX_TIME_DAY = DAY_PER_YEAR * 100  # 环境最多运行的天数
REWARD_GAMMA = 0.99  # Exponentially Weighted Moving Average（指数加权滑动平均）方式计算 reward（奖赏）之和时所用的参数 gamma

DEFAULT_HP = 100  # 每条公路的起始健康度
DEFAULT_TRAFFIC = 100  # 默认起始每条公路的每日车流量
DEFAULT_STATUS = "keep"  # 每条公路的默认状态
DEFAULT_REMAIN = math.inf  # 默认状态的持续时间

# 用字典代表每一条公路
HIGHWAYS = [
    {"name": "S1", "total_len": 20, "hp": 100, "traffic": 200},
    {"name": "S2", "total_len": 40}
]


def DEFAULT_NEW_TRAFFIC(time_remain: float, old_traffic: float):
    return old_traffic


""" 所有可能的动作
new_hp:
    一个函数，计算采取动作后的新的hp值。
    输入：旧hp值，输出：新hp值
cost:
    采取动作所用的花销
time:
    该动作的持续时间
traffic:
    一个函数，计算动作持续期间的每日车流量，每隔单位时间会调用一次
    输入：状态剩余时间，旧车流量值
    输出：新车流量值
"""
ACTIONS = [
    {"name": "small", "new_hp": lambda x: x + 10, "cost": 50000, "time": 1, "traffic": lambda tm, tfc: 0},
    {"name": "big", "new_hp": lambda x: x + 30, "cost": 200000, "time": 15, "traffic": lambda tm, tfc: 0},
    {"name": "remake", "new_hp": lambda x: 100, "cost": 500000, "time": 60, "traffic": lambda tm, tfc: 0},
    {"name": "keep", "new_hp": lambda x: x, "cost": 0, "time": math.inf, "traffic": lambda tm, tfc: tfc}
]

""" 突发事件
new_hp:
    一个函数，计算突发事件发生后的新的hp值。
    输入：旧hp值，输出：新hp值
frequency：
    每日发生概率
"""
EMERGENCIES = [
    {"name": "destroy", "new_hp": lambda x: 0, "frequency": 1 / DAY_PER_YEAR / 100, "type": "Poisson"},
    {"name": "damage", "new_hp": lambda x: x / 2, "frequency": 1 / DAY_PER_YEAR / 10, "type": "Poisson"}
]


def reward(traffic, cost, hp):
    """ 单步奖赏计算函数

    Parameters
    ----------
    traffic:
        车流量
    cost:
        开销
    hp:
        健康度

    Returns
    -------

    """
    income = traffic * 300
    return income + cost - 100 * hp


def traffic_to_aging_speed(traffic: float) -> float:
    """ 由车流量计算健康度损耗

    Parameters
    ----------
    traffic : float
        car number each day

    Returns
    -------
    aging_speed : float
        new_hp = old_hp - aging_speed
    """
    C = 1.49  # 换算系数
    N = C * traffic  # 轴次
    N *= DAY_PER_YEAR
    aging = 1.164 + 6.621 * 10 ** (-7) * N + 3.381 * 10 ** (-14) * N * N  # 断面 PQI 降低值
    aging /= DAY_PER_YEAR
    return aging
