"""难度分 -> 每样本保留率的映射（分位数校准，保证平均预算=目标预算）。"""
import numpy as np


class BudgetMapper:
    """在校准集难度分上拟合分位数映射，推理时给出每样本 keep_ratio。

    保证（校准集上）mean(ratio) == target_ratio，与固定比例公平比较。
    """

    def __init__(self, r_min: float = 0.02, r_max: float = 0.5):
        self.r_min, self.r_max = r_min, r_max
        self.calib_scores = None
        self.scale = 1.0

    def fit(self, scores, target_ratio: float):
        self.calib_scores = np.sort(np.asarray(scores))
        # 线性映射: 分位数 q -> r_min + q*(r_max-r_min)，再缩放使均值命中目标
        q = np.linspace(0, 1, len(self.calib_scores))
        raw = self.r_min + q * (self.r_max - self.r_min)
        self.scale = target_ratio / raw.mean()
        return self

    def ratio(self, score: float) -> float:
        q = np.searchsorted(self.calib_scores, score) / len(self.calib_scores)
        r = (self.r_min + q * (self.r_max - self.r_min)) * self.scale
        return float(np.clip(r, self.r_min, self.r_max))
