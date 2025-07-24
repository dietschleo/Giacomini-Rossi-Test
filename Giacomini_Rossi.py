"""
Based on the work of Giacomini, R. and B. Rossi, “Forecast Comparisons in Unstable Environments”. Journal of Applied Econometrics 25(4), April 2010, 595-620.
Original Stata code available at:
https://sites.google.com/site/barbararossiwebsite/codes/stata-codes
http://fmwww.bc.edu/repec/bocode/g/giacross.ado
"""
import numpy as np #numpy==1.26.4
import pandas as pd
from statsmodels.tsa.stattools import acovf #statsmodels==0.14.1

def fluctuation_test(loss1, loss2, mu=0.5, lag_truncate=0, conf_level=0.05, side=2):
    """
    Giacomini-Rossi Fluctuation Test for forecast stability
    Parameters:
    -----------
    loss1, loss2: arrays of forecast losses (e.g., squared errors)
    mu: rolling window fraction (0.1 to 0.9 in steps of 0.1)
        Controls the size of rolling sub-samples as fraction of total sample.
        mu=0.5 means each rolling window contains 50% of total observations.
        Smaller mu = more rolling windows but less power per window.
    lag_truncate: bandwidth for HAC variance estimation (Bartlett kernel)
    conf_level: significance level (0.05 or 0.10)
    side: 1 = one-sided test, 2 = two-sided test
    """

    d = loss1 - loss2
    T = len(d)
    window = int(mu * T)
    if window <= 1:
        raise ValueError("Window too small")

    # 1) Roll DM statistics
    dm_stats = np.empty(T - window + 1)
    for t in range(window, T + 1):
        dt = d[t - window:t]
        mean_dt = dt.mean()
        # HAC variance
        acovs = acovf(
            dt,
            nlag=lag_truncate,     # limit to desired lag count
            demean=True,           # subtract mean
            adjusted=False,        # unbiased default
            fft=True,              # use FFT for speed
            missing='none'         # handle NaNs appropriately
        )

        var_hac = acovs[0] + 2 * acovs[1:].sum()
        dm_stats[t - window] = mean_dt / np.sqrt(var_hac / window)

    # 2) Critical value lookup approximated via Monte Carlo simulations or precomputed
    cv = _lookup_critical_value(mu, conf_level, two_sided=(side == 2))

    # 3) Collect results
    idx = np.arange(window, T + 1)
    return pd.DataFrame({
        'time_idx': idx,
        'dm_stat': dm_stats,
        'cv_low': -cv if side == 2 else None,
        'cv_high': cv,
    })

def _lookup_critical_value(mu, alpha, two_sided=True):
    """
    Retrieve critical value for Giacomini-Rossi Fluctuation Test
    Parameters:
        mu: float ∈ {0.1, 0.2, ..., 0.9}
        alpha: significance level (0.05 or 0.10)
        two_sided: if True, use 2-sided critical values
    Returns:
        critical value (float)
    """
    # Sanitize and round mu to nearest 0.1 (as in Stata)
    mu = round(mu + 1e-8, 1)
    if mu not in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        raise ValueError(f"mu must be a rolling window fraction in {{0.1, 0.2, ..., 0.9}}, got {mu}. "
                         f"This determines the size of each rolling sub-sample as a fraction of total sample size.")

    if alpha not in [0.05, 0.10]:
        raise ValueError("alpha must be 0.05 or 0.10")

    # Lookup tables from the giacross.ado matrices
    if two_sided:
        table = {
            0.1: {0.05: 3.393, 0.10: 3.170},
            0.2: {0.05: 3.179, 0.10: 2.948},
            0.3: {0.05: 3.012, 0.10: 2.766},
            0.4: {0.05: 2.890, 0.10: 2.626},
            0.5: {0.05: 2.779, 0.10: 2.500},
            0.6: {0.05: 2.634, 0.10: 2.356},
            0.7: {0.05: 2.560, 0.10: 2.252},
            0.8: {0.05: 2.433, 0.10: 2.130},
            0.9: {0.05: 2.248, 0.10: 1.950}
        }
    else:
        table = {
            0.1: {0.05: 3.176, 0.10: 2.928},
            0.2: {0.05: 2.938, 0.10: 2.676},
            0.3: {0.05: 2.770, 0.10: 2.428},
            0.4: {0.05: 2.624, 0.10: 2.334},
            0.5: {0.05: 2.475, 0.10: 2.168},
            0.6: {0.05: 2.352, 0.10: 2.030},
            0.7: {0.05: 2.248, 0.10: 1.904},
            0.8: {0.05: 2.080, 0.10: 1.740},
            0.9: {0.05: 1.975, 0.10: 1.600}
        }

    return table[mu][alpha]