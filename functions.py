import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from iblatlas.regions import BrainRegions


def get_spikes_for_cluster(spike_clusters, spike_times, cluster):
    # requires that spike_times and spike_clusters are sorted
    start_ix, stop_ix = np.searchsorted(spike_clusters, [cluster, cluster + 1])
    return np.sort(spike_times[start_ix:stop_ix])


def expon_decay(t, A, tau, b):
    return A * np.exp(-t / tau) + b


def bin_spike_train(t: np.ndarray, dt: float):
    t_start = 0
    t_stop = t[-1]
    bins = np.arange(t_start, t_stop + dt, dt)
    ix = np.digitize(t, bins)
    tb = np.zeros_like(bins)
    np.add.at(tb, ix, 1)
    return tb


def calc_acorr(t: np.ndarray, dt=0.005, n_lags=500, n_offset=10):
    # for a spike train t
    # first binarize with bin size dt
    tb = bin_spike_train(t, dt)

    # calculate autocorrelation
    # skip n_offset bins, n_offset * dt have to be larger than the
    # refractory period dip for the exp decay fit to make sense
    lags = np.arange(n_offset, n_lags)
    acorr = np.array([np.sum(tb * np.roll(tb, lag)) for lag in lags])

    return acorr, lags


def fit_acorr(acorr, dt, lags):
    # start form shoulder after refractory period
    max_ix = np.argmax(acorr)

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
        return (np.nan, np.nan, np.nan)


def get_spikes(pid, one):
    # convert to probe name and eid
    eid, pname = one.pid2eid(pid)

    # get spike times
    spike_times = one.load_dataset(
        eid, "spikes.times", collection=f"alf/{pname}/pykilosort"
    )

    # get spike clusters
    spike_clusters = one.load_dataset(
        eid, "spikes.clusters", collection=f"alf/{pname}/pykilosort"
    )
    return spike_times, spike_clusters


def get_clusters_metrics(pid, one):
    # convert to probe name and eid
    eid, pname = one.pid2eid(pid)

    # get channel for each cluster
    clusters_channels = one.load_dataset(
        eid, f"alf/{pname}/pykilosort/clusters.channels.npy"
    )

    # forming the unit dataframe
    clusters_metrics = one.load_dataset(
        eid, f"alf/{pname}/pykilosort/clusters.metrics.pqt"
    )

    # get and add the uuid for each cluster
    uuids = one.load_dataset(eid, f"alf/{pname}/pykilosort/clusters.uuids.csv")
    clusters_metrics = clusters_metrics.join(uuids)

    # combining with histological / location information
    # which channel at which brain area (allen id)
    channel_allen_ids = one.load_dataset(
        eid, f"alf/{pname}/pykilosort/channels.brainLocationIds_ccf_2017.npy"
    )

    # converting allen id in to allen acronym
    brain_regions = BrainRegions()
    channel_allen_acro = brain_regions.id2acronym(channel_allen_ids)

    # mapping allen_acro onto spike_cluster
    clusters_metrics["allen_acro"] = channel_allen_acro[clusters_channels]

    return clusters_metrics
