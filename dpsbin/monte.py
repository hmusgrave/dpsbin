import numpy as np
from random import randint

def pmf(M, T, trial_count=10_000):
    """Computes chance of stopping after k rolls of 1dM trying to reach total T

    Args:
        M: positive integer, die roll from 1 -> M (inclusive)
        T: non-negative integer, total we want to meet or exceed

    Kwargs:
        trial_count: positive integer (default 10_000), number of
        simulations used to estimate result probabilities.
        Bigger values give better results but take more time.

    Returns:
        An array of probabilities with shape (T+1,)
        rtn[k] == <chance of stopping after roll k>

    Notes:
        Argument validation is the responsibility of the caller.

    Design:
        This is intended to be a reference implementation to which
        other methods whose correctness is non-obvious can be compared.
        The use of sequence accelerators and other such techniques should
        be relegated to other implementations.
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
