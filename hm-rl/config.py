import math

UNIT_LEN_KM = 10
TIME_INTERVAL_DAY = 1
DAY_PER_YEAR = 365
MAX_TIME_DAY = DAY_PER_YEAR * 100
REWARD_GAMMA = 0.99

DEFAULT_HP = 100
DEFAULT_AGING = 0.1
DEFAULT_TRAFFIC = 100
DEFAULT_STATUS = "keep"
DEFAULT_REMAIN = math.inf

HIGHWAYS = [
    {"name": "S1", "total_len": 20, "hp": 100, "traffic": 200},
    {"name": "S2", "total_len": 40}
]


def DEFAULT_NEW_TRAFFIC(time_remain: float, old_traffic: float):
    return old_traffic


ACTIONS = [
    {"name": "small", "new_hp": lambda x: x + 10, "cost": 50000, "time": 1, "traffic": lambda tm, tfc: 0},
    {"name": "big", "new_hp": lambda x: x + 30, "cost": 200000, "time": 15, "traffic": lambda tm, tfc: 0},
    {"name": "remake", "new_hp": lambda x: 100, "cost": 500000, "time": 60, "traffic": lambda tm, tfc: 0},
    {"name": "keep", "new_hp": lambda x: x, "cost": 0, "time": math.inf, "traffic": lambda tm, tfc: tfc}
]

EMERGENCIES = [
    {"name": "destroy", "new_hp": lambda x: 0, "frequency": 1 / DAY_PER_YEAR / 100, "type": "Poisson"},
    {"name": "damage", "new_hp": lambda x: x / 2, "frequency": 1 / DAY_PER_YEAR / 10, "type": "Poisson"}
]


def reward(traffic, cost, hp):
    income = traffic * 300
    return income + cost - 100 * hp


def traffic_to_aging_speed(traffic: float) -> float:
    """

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
