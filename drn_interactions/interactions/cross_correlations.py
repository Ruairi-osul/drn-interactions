from scipy.signal import correlate
import numpy as np


def ccovf(x, y, adjusted=True, demean=True, fft=True):
    n = (len(x) * 2) - 1
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if adjusted:
        d = np.arange(n, 0, -1)
    else:
        d = n

    method = "fft" if fft else "direct"
    lags = np.arange(-len(x) + 1, len(y))
    return lags, correlate(xo, yo, "full", method=method) / d


def ccf(x, y, adjusted=True, fft=True):
    lags, cvf = ccovf(x, y, adjusted=adjusted, demean=True, fft=fft)
    return lags, cvf / (np.std(x) * np.std(y))
