# dpsbin

Computes the probability mass function for an extended negative binomial distribution essential for accurate DPS calculations in a variety of turn/tick-based games.


# Description

Common methods for estimating how long it takes to finish off an opponent in typical hack-and-slash games are inaccurate due to factors like overkill. Those crude calculations are often good enough, but in case they're not you can turn to this library to get an exact answer.

# Installation

dpsbin requires Python 3 or greater and will be available on PyPI when the API is stable. Till then, it can be cloned and used directly from the repo (in an environment with numpy installed) or else installed locally with pip.

```
cd dpsbin
python -m pip install -e .
```

# Using dpsbin

The only notable entry point is `dpsbin.pmf`. Suppose you want to know how many times you'll have to roll a 6-sided die to reach a total of 40 or more.

```
import dpsbin, numpy as np, math

probs = dpsbin.pmf(6, 40)

# probs[k] is the probability you need to roll k times.
assert math.isclose(np.sum(probs), 1)

# easily find out how many rolls you'll need on average.
avg = np.dot(probs, np.arange(len(probs)))
```