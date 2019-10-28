import pandas as pd
import numpy as np


def sample_weights(treatments, propensities):
    def cap(x):
        if np.isclose(x[1], 0, atol=1e-3):
            x[1] = 0.05
        elif np.isclose(x[1], 1, atol=1e-3):
            x[1] = 0.95
        return x

    def weight(x):
        x = cap(x)
        return 1 / (1 - x[1]) if x[0] == 1 else 1 / x[1]

    df = pd.concat([treatments, propensities], axis=1)
    return df.apply(lambda x: weight(x), axis=1) 