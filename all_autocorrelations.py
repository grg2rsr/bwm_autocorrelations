# %% imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging

# local functions
import functions

# %% ONE instantiation
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE

ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
cache_dir = Path("/home/georg/ibl_scratch/tmp_cache")
one = ONE(password="international", mode="remote", cache_dir=cache_dir)

# suppress ONE FutureWarnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# %% parameters setup
run_name = "new_testing"
recalculate = True
dt = 0.005
n_lags = 500
min_presence_ratio = 0.7
min_firing_rate = 0.5

output_folder = Path(f"/home/georg/ibl_scratch/bwm_autocorrelations/{run_name}")
output_folder.mkdir(exist_ok=True, parents=True)

# create logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(output_folder / f"{run_name}.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger()

# %% main loop
# for brain_region in brain_regions:
# pids = list(one.search_insertions(atlas_acronym=brain_region))
# logger.info(f"found {len(pids)} recordings for brain region {brain_region}")

with open("bwm_pids", "r") as fH:
    pids = [pid.strip() for pid in fH.readlines()]

for pid in tqdm(pids[:5]):
    # avoid duplicate calculations
    output_file = output_folder / f"{pid}.csv"
    if not recalculate:
        if output_file.exists():
            logger.info(f"skipping {pid} because it has already been processed")
            continue

    # run calculation for this pid
    logger.info(f"processing {pid}")

    # load spike sorting data
    sl = SpikeSortingLoader(one, pid=pid)
    spikes, clusters, channels = sl.load_spike_sorting()

    clusters_metrics = clusters["metrics"]
    clusters_metrics["uuid"] = clusters["uuids"]
    clusters_metrics["pid"] = pid

    # map each cluster to brain regions
    clusters_metrics["brain_region"] = channels["acronym"][clusters["channels"]]

    # qc subselect presence ratio
    ix = clusters_metrics["presence_ratio"] > min_presence_ratio
    clusters_metrics = clusters_metrics.loc[ix]

    # qc subselect by firing rate
    ix = clusters_metrics["firing_rate"] > min_firing_rate
    clusters_metrics = clusters_metrics.loc[ix]

    # calculate autocorrelation for all units
    for i in clusters_metrics.cluster_id:
        # note: direct indexing instead of np.searchsorted is faster here
        t = spikes["times"][spikes["clusters"] == i]
        acorr, lags = functions.calc_acorr(t, dt, n_lags)
        popt = functions.fit_acorr(acorr, dt, lags)
        clusters_metrics.loc[i, "tau"] = popt[1]

    # store
    clusters_metrics.to_csv(output_file)
