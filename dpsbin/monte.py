import numpy as np
from random import randint

def pmf(M, T, trial_count=20000):
    """See exact.pmf for details.

    TODO: Flesh out docstring
    """
    def simulate():
        t, n = 0, 0
        while t < T:
            t, n = t+randint(1, M), n+1
        return n

    rtn = np.zeros(T+1)
    for _ in range(trial_count):
        rtn[simulate()] += 1

    return rtn/np.sum(rtn)
