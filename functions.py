import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# from iblatlas.regions import BrainRegions
# from brainbox.io.one import SpikeSortingLoader

# abandoned this strategy, see benchmarking script in ibldevtools/georg
# def get_spikes_for_cluster(spike_clusters, spike_times, cluster):
#     # requires that spike_times and spike_clusters are sorted
#     start_ix, stop_ix = np.searchsorted(spike_clusters, [cluster, cluster + 1])
#     return np.sort(spike_times[start_ix:stop_ix])


def expon_decay(t, A, tau, b):
    return A * np.exp(-t / tau) + b


def bin_spike_train(t: np.ndarray, dt: float, t_start=None, t_stop=None):
    t_start = 0 if t_start is None else t_start
    t_stop = t[-1] if t_stop is None else t_stop
    bins = np.arange(t_start, t_stop + dt, dt)
    ix = np.digitize(t, bins)
    tb = np.zeros_like(bins)
    np.add.at(tb, ix, 1)
    return tb


def calc_acorr(t: np.ndarray, dt=0.005, n_lags=500):
    # for a spike train t
    # binarize
    tb = bin_spike_train(t, dt)

    # calculate autocorrelation
    lags = np.arange(0, n_lags)
    acorr = np.array([np.sum(tb * np.roll(tb, lag)) for lag in lags])
    return acorr, lags


def fit_acorr(acorr, dt, lags, n_offset=1):
    # start form shoulder after refractory period
    # skip n_offset lags for finding the peak
    max_ix = np.argmax(acorr[n_offset:])

    # fit an exponential decay to the autocorrelation
    n_lags = lags.shape[0]
    p0 = (acorr[0], n_lags / 2 * dt, acorr[-1])
    eps = 1e-10
    bounds = ((0, eps, 0), (np.inf, 100, np.inf))
    try:
        popt = curve_fit(
            expon_decay, lags[max_ix:] * dt, acorr[max_ix:], p0, bounds=bounds
        )[0]
        return popt
    except RuntimeError:
        return (np.nan, np.nan, np.nan)  #     return clusters_metrics
