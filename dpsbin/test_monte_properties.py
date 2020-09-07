from dpsbin import _monte as monte
import numpy as np, pytest, math

@pytest.mark.parametrize('M, T', [[M, T]
    for M in range(1, 4)
    for T in range(4)
])
def test_is_pma(M, T):
    pma = monte.pmf(M, T)
    assert len(pma)==T+1
    assert math.isclose(np.sum(pma), 1.)
    assert np.min(pma)>=0.
