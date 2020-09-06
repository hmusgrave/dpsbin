from dpsbin import exact, monte
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

    # TODO: This is pretty crude
    assert np.linalg.norm(a-b) < 2e-2
