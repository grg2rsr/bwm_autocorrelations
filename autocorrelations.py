# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# %% just local plotting
# %matplotlib qt5
# mpl.rcParams["figure.dpi"] = 330

# %% ONE instantiation
from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound

ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
one = ONE(password="international", mode="remote")

output_folder = Path("/home/georg/ibl_scratch/bwm_autocorrelations")
output_folder.mkdir(exist_ok=True, parents=True)

# parameters for this run
dt = 0.005
n_lags = 500
min_presence_ratio = 0.7
min_firing_rate = 0.5

# %%
import functions
import sys

sys.path.append(
    "/home/georg/code/ibldevtools/georg/bwm_autocorrelations"
)  # I clearly don't know what I am doing ...

brain_regions = ["ORB"]

for brain_region in brain_regions:
    pids = list(one.search_insertions(atlas_acronym=brain_region))

    for pid in tqdm(pids):
        # avoid duplicate calculations
        output_file = output_folder / f"{brain_region}_{pid}.csv"
        if output_file.exists():
            continue
        try:
            clusters_metrics = functions.get_clusters_metrics(pid, one)
        except ALFObjectNotFound:
            print(f"alf object not found for {pid}")
            continue

        clusters_metrics["pid"] = pid

        # subselect to brain region
        ix = [c.startswith(brain_region) for c in clusters_metrics["allen_acro"]]
        clusters_metrics = clusters_metrics.loc[ix]

        ix = clusters_metrics["presence_ratio"] > min_presence_ratio
        clusters_metrics = clusters_metrics.loc[ix]

        ix = clusters_metrics["firing_rate"] > min_firing_rate
        clusters_metrics = clusters_metrics.loc[ix]

        # load spike sorting data
        spike_times, spike_clusters = functions.get_spikes(pid, one)

        order = np.argsort(spike_clusters)
        spike_times = spike_times[order]
        spike_clusters = spike_clusters[order]

        # form selection
        for i in clusters_metrics.cluster_id:
            t = functions.get_spikes_for_cluster(spike_clusters, spike_times, i)
            acorr, lags = functions.calc_acorr(t, dt, n_lags)
            popt = functions.fit_acorr(acorr, dt, lags)
            clusters_metrics.loc[i, "tau"] = popt[1]

        clusters_metrics.to_csv(output_file)
