from dpsbin import _exact as exact
import pytest, numpy as np

@pytest.mark.parametrize('M', [[x] for x in range(5)])
def test_zero_total(M):
    # When the total is 0, we always achieve it after 0 rolls
    x = exact.pmf(M, 0)
    assert len(x)==1
    assert x[0]==1

@pytest.mark.parametrize('M, T, p', [L+[p] for L in [
    [1, 0],
    [2, 0],
    [1, 1],
    [1, 2],
    [3, 5]
] for p in [0., .3, .5, .9, 1.]])
def test_zero_rolls(M, T, p):
    # No matter the pma, when we ask for data for at most 0 rolls
    # we only need the first value of the pma
    pma = exact.pmf(M, T)
    res = exact.with_zeros(0, p, pma)
    assert len(res)==1
    assert res[0]==pma[0]

@pytest.mark.parametrize('pma, k', [
    [np.array([1., 0.]), 3],
    [np.array([1.]), 4],
    [np.array([1.]), 0],
    [np.array([1.]), 1],
])
def test_nonzero_prob_zero_rolls(pma, k):
    # If a pma indicates that the goal can be achieved with 0 rolls,
    # then without interference beyond the scope of the single random
    # variable being analyzed the only possibility is that the goal
    # can always be achieved in exactly 0 rolls.
    res = exact.with_zeros(k, .5, pma)
    assert len(res)==(k+1)
    assert res[0]==1
    assert np.sum(res)==1
    assert np.min(res)>=0

@pytest.mark.parametrize('M, T, p, k', [
    [M, T, p, k] for M in range(1, 4)
    for T in range(3)
    for p in [0., .3, .5, .9, 1.]
    for k in range(4)
])
def test_with_zeros_has_correct_length(M, T, p, k):
    pma = exact.pmf(M, T)
    z = exact.with_zeros(k, p, pma)
    assert len(z)==(k+1)
