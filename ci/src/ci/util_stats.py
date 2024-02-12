import numpy as np
import pandas as pd


def get_standard_error(y: pd.Series):
    return y.std() / np.sqrt(len(y))


def get_confidence_interval(y: pd.Series, c: float = 1.96):
    return (y.mean() - c * get_standard_error(y), y.mean() + c * get_standard_error(y))
