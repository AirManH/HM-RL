import math

TIME_INTERVAL_DAY = 1  # 时间片
DAY_PER_YEAR = 365  # 一年的天数
MAX_TIME_DAY = DAY_PER_YEAR * 100  # 环境最多运行的天数
REWARD_GAMMA = 0.99  # Exponentially Weighted Moving Average（指数加权滑动平均）方式计算 reward（奖赏）之和时所用的参数 gamma

DEFAULT_HP = 100  # 每条公路的起始健康度
DEFAULT_TRAFFIC = 25000  # 默认起始每条公路的每日车流量
DEFAULT_STATUS = {
    "name": "in_use",
    "remain": math.inf,
    "stoppable": True
}  # 每条公路的默认状态

# 用字典代表每一条公路
HIGHWAYS = [
    {"name": "GF", "traffic": int(26575444 / DAY_PER_YEAR / 2), "toll": int(190837600 / 26575444), "note": "广佛高速公路"},
    {"name": "FK", "traffic": int(23968177 / DAY_PER_YEAR / 2), "toll": int(569280800 / 23968177), "note": "佛开高速公路"},
    {"name": "JG", "traffic": int(30170232 / DAY_PER_YEAR / 2), "toll": int(541588200 / 30170232), "note": "京珠高速公路广珠段"},
    {"name": "GH", "traffic": int(21450514 / DAY_PER_YEAR / 2), "toll": int(800781100 / 21450514), "note": "广惠高速公路"},
    {"name": "HY", "traffic": int(17044547 / DAY_PER_YEAR / 2), "toll": int(106159000 / 17044547), "note": "惠盐高速公路"},
    {"name": "GZ", "traffic": int(14315677 / DAY_PER_YEAR / 2), "toll": int(289978900 / 14315677), "note": "广肇高速公路"},
    {"name": "JZ", "traffic": int(20429959 / DAY_PER_YEAR / 2), "toll": int(196259200 / 20429959), "note": "江中高速公路"},
    {"name": "KD", "traffic": int(1187605 / DAY_PER_YEAR / 2), "toll": int(115537300 / 1187605), "note": "康大高速"},
    {"name": "GK", "traffic": int(1158072 / DAY_PER_YEAR / 2), "toll": int(73247800 / 1158072), "note": "赣康高速"},
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
new_traffic:
    一个函数，计算动作持续期间的每日车流量，每隔单位时间会调用一次
    输入：状态剩余时间，旧车流量值
    输出：新车流量值
"""
ACTIONS = [
    {"name": "small", "cost": 50000, "time": 2,
     "new_hp": lambda x: x + 10, "new_traffic": lambda tm, tfc: 0,
     "stoppable": False},
    {"name": "big", "cost": 200000, "time": 20,
     "new_hp": lambda x: x + 10, "new_traffic": lambda tm, tfc: 0,
     "stoppable": False},
    {"name": "remake", "cost": 500000, "time": 60,
     "new_hp": lambda x: 100, "new_traffic": lambda tm, tfc: 0,
     "stoppable": False},
    {"name": "in_use", "cost": 0, "time": math.inf,
     "new_hp": lambda x: x, "new_traffic": lambda tm, tfc: tfc,
     "stoppable": True}
]

""" 突发事件
new_hp:
    一个函数，计算突发事件发生后的新的hp值。
    输入：旧hp值，输出：新hp值
frequency：
    每日发生概率
"""
EMERGENCIES = [
    {"name": "destroy", "cost": 0, "time": 10,
     "new_hp": lambda x: x - 20, "frequency": 1 / DAY_PER_YEAR / 100,
     "stoppable": False},
    {"name": "damage", "cost": 0, "time": 3,
     "new_hp": lambda x: x - 5, "frequency": 1 / DAY_PER_YEAR / 10,
     "stoppable": False},
    {"name": "damage", "cost": 0, "time": 1,
     "new_hp": lambda x: x - 1, "frequency": 1 / DAY_PER_YEAR / 1,
     "stoppable": False}
]


def reward(traffic: int,
           toll: float,
           hp: float,
           cost: float):
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
    # 平均hp大约每日-0.1
    income = traffic * toll
    return (income - cost) / 10000 + 10 * hp - 1000


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

    AADTT, DDF, LDF = traffic, 0.5, 0.4
    EALF_parameters = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    EALF_m = [a * b + c * d for a, b, c, d in EALF_parameters]
    N = AADTT * DDF * LDF + sum(EALF_m)  # 轴次

    N = 1.49 * traffic

    N *= DAY_PER_YEAR

    aging = 1.164 + 6.621 * 10 ** (-7) * N + 3.381 * 10 ** (-14) * N * N  # 断面 PQI 降低值
    aging /= DAY_PER_YEAR

    return aging
