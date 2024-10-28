# %% imports
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# %% plot config
mpl.rcParams['figure.dpi'] = 166
%matplotlib qt5

# %% get data
output_folder = Path("/home/georg/ibl_scratch/bwm_autocorrelations")
csv_files = [f for f in list(output_folder.iterdir()) if f.suffix == '.csv']
df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], axis=0)
df = df.reset_index()
df.loc[df['tau'] > 50, 'tau'] = np.nan

# %%
axes = sns.barplot(df, x='allen_acro', y='tau')
axes.set_title(df.shape[0])
sns.despine(axes.figure)

# %%
ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
one = ONE(password="international", mode="remote")

# %% inspecting a single one
row = df[df['tau'] > 5].iloc[0]
pid = row.pid
uuid = row.uuids

# %% 
dt = 0.005
n_lags = 500

import functions
clusters_metrics = functions.get_clusters_metrics(pid, one)
cluster_id = clusters_metrics.loc[clusters_metrics['uuids'] == uuid]['cluster_id'].values[0]

spike_times, spike_clusters = functions.get_spikes(pid, one)

order = np.argsort(spike_clusters)
spike_times = spike_times[order]
spike_clusters = spike_clusters[order]

t = functions.get_spikes_for_cluster(spike_clusters, spike_times, cluster_id)
acorr, lags = functions.calc_acorr(t, dt, n_lags)
popt = functions.fit_acorr(acorr, dt, lags)

# %%
fig, axes = plt.subplots()
axes.plot(lags*dt, acorr)
axes.plot(lags*dt, functions.expon_decay(lags*dt, *popt))

# %%
