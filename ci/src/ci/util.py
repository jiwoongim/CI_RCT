import math
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RNG = np.random.RandomState()

DF = pd.core.frame.DataFrame
ITEM = 0

from ci.rct_example.online_classroom import OnlineClassRoom


def get_dataset(path: str) -> DF:
    #'online_classroom' in name:
    return OnlineClassRoom(pd.read_csv(f"{path}.csv"))


MEAN = 0
SE = 1


class Bookkeeping:
    def __init__(self, num_action: int):
        self.num_action = num_action
        self.cos = defaultdict(list)
        self.pos = defaultdict(list)
        self.ate = defaultdict(list)

    def add_conditional_outcomes(self, epoch, cos_stat):
        self.cos["epoch"].append(epoch)
        for action in range(self.num_action):
            self.cos[action].append(cos_stat[action])

    def add_potential_outcomes(self, epoch, pos_stat):
        self.pos["epoch"].append(epoch)
        for action in range(self.num_action):
            self.pos[action].append(pos_stat[action])

    def add_ate(self, epoch, ate_stat):
        self.ate["epoch"].append(epoch)
        self.ate["ate"].append(ate_stat)

    def get_stats(self):
        return self.pos, self.ate

    def save_figure(self, fname, window=10, vrange=(-0.5, 0.5)):
        ATE = 0
        XA = self.ate["epoch"]
        fig, axs = plt.subplots(1, self.num_action + 1, figsize=(15, 7))

        ate_stat = np.asarray(self.ate["ate"])
        mean, se = ate_stat[:, MEAN], ate_stat[:, SE]
        axs[ATE].fill_between(XA, mean - se, mean + se, alpha=0.25, color="red")
        axs[ATE].plot(XA, mean, color="red")
        axs[ATE].set_title("Average Treatment Effect")

        XP = self.pos["epoch"]
        XC = self.cos["epoch"]
        for action in range(self.num_action):
            i = action + 1
            pos_np = np.asarray(self.pos[action])
            mean = pos_np[:, MEAN]
            se = pos_np[:, SE]
            axs[i].fill_between(XP, mean - se, mean + se, alpha=0.25, color="blue")
            axs[i].plot(XP, mean, color="blue", label="Potential Outcomes")
            axs[i].plot(
                XC, self.cos[action], color="green", label="Conditional Outcomes"
            )
            axs[i].set_title("Potential Outcomes")
            # axs[i].set_ylim(vrange)
            axs[i].legend()
        plt.xlabel("Iteration")
        plt.tight_layout()
        plt.savefig(f"figs/{fname}.pdf")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sample_bernoulli(prob: float) -> tuple[int, float]:
    sample = RNG.uniform(low=0, high=1, size=1).flatten()[ITEM]
    if sample < prob:
        return 0, 1 - prob
    return 1, prob
