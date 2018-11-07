from pandas import Series
from numpy import random, histogram, digitize, arange, sqrt, array


def jitter_offset(x, nbins=100, multiplier=2.0):
    bins, edges = histogram(x, bins=nbins, density=True)
    bins /= bins.max()
    try:
        assignments = digitize(x, edges, right=True) - 1
        assignments[assignments < 0] = 0
        output = sqrt(Series(bins).iloc[assignments])
    except ValueError:
        output = Series([0] * x.shape[0])
    output.index = x.index
    output *= random.choice([-multiplier, multiplier], output.shape[0])
    output *= random.choice(arange(1, 25) / 100., output.shape[0])
    return output
