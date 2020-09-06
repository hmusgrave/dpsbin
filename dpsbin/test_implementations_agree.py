from dpsbin import exact, monte
from scipy.stats import wasserstein_distance
import pytest, numpy as np

@pytest.mark.parametrize('M, T', [
    [1, 1],
    [1, 2],
    [1, 3],
    [2, 1],
    [2, 2],
    [2, 3],
    [5, 3],
    [5, 6],
    [5, 7],
])
def test_exact_matches_simulation(M, T):
    a = exact.pmf(M, T)
    b = monte.pmf(M, T)

    assert wasserstein_distance(a, b) < 1e-2
