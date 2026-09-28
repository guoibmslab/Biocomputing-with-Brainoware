#!/usr/bin/python
# -------------------------------------------------------------
# Author: Hongwei Cai, Huiyu Chu
# Date:   2025-06-06
# Description: Biocomputing with Brainoware
#              Procedure 2 - Reservoir computing hardware properties
#              1. Measuring evoked responses
#              This script describes how to read saved spike and stimulation timing files,
#              and plot the PSTH (Peri-Stimulus Time Histogram) of evoked responses. This script
#              only plots the condition of 500us, 500mV stimulation response as an example.
# -------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spykes.plot.neurovis import NeuroVis
from spykes.plot.popvis import PopVis

## read spike and stimulation timing files ##
path = os.path.dirname(os.path.abspath(__file__)) + os.sep
filename = "fig3a_psth_exp"
spike_time = pd.read_csv(
    path + filename + "_spike.csv", dtype=object, usecols=[1, 2, 3]
)
spike_time = spike_time.astype({"time": "float"})
#
filename_stim = filename + "_sti_time"
sti_time = pd.read_csv(path + filename_stim + ".csv", dtype=float, header=None)
sti_time = sti_time.values.tolist()
sti_time = np.concatenate(sti_time)

## hyper parameters ##
top_n = 16  # top N active electrode to be selected
repetitions_per_pulse = 5  # repetitions of each stimulation parameter
amp_max = 700  # maximum stimulation amplitude
amp_init = 100  # initial stimulation amplitude
sti_amp = np.arange(100, amp_max + amp_init, amp_init)  # list of all stim amps
sti_dura = [100, 200, 300, 400, 500]  # list of all stimulation durations
durations = np.size(sti_dura)  # number of durations
amplitudes = np.size(sti_amp)  # number of amplitudes

## filter out stimulation artifacts happening between [-2ms,+2ms] of each stimulation time point ##
for sti_t in sti_time:
    spike_time = spike_time[
        (spike_time["time"] < sti_t - 0.002)
        | (spike_time["time"] > sti_t + 0.002)
    ]
ch_0 = spike_time["channel"].value_counts()[:top_n].index.tolist()
spike_time = spike_time.astype({"time": "string"})
spike_time_filtered = spike_time[
    spike_time[["channel", "time"]]
    .apply(lambda x: x.str.contains("|".join(ch_0)))
    .any(axis=1)
]
spike_time_filtered = spike_time_filtered.astype({"time": "float"})
spike_time_filtered = spike_time_filtered.astype({"amplitude": "float"})
## re-organize spike information as a list per active electrode (channel) ##
spike_time_filtered_per_channel = spike_time_filtered.groupby(
    "channel", as_index=False
).agg(lambda x: x.tolist())


## initiate spykes NeuroVis and PopVis object with spike_time_filtered_per_channel ##
def initiate_neurons(raw_data):
    neuron_list = list()

    for i in range((raw_data["time"]).shape[0]):
        spike_times = raw_data["time"][i]

        # instantiate neuron
        neuron = NeuroVis(spike_times, name="Electrode %d" % (i + 1))
        neuron_list.append(neuron)

    return neuron_list


neuron_list = initiate_neurons(spike_time_filtered_per_channel)[
    :top_n
]  # <-- change top_n value if more or less electrodes are needed.
pop = PopVis(neuron_list)

## construct stimulation dataframe to select stimulation timing information for ploting ##
# here choose 500mV amplitude and 500us phase duration to display
event = "stiTime"
condition = "Sequence"
window = [-50, 810]
binsize = 50
stimulation_df = pd.DataFrame()
dura_num = 4
dura = sti_dura[dura_num]  # 500us
amp_plot = [4]  # fourth amp, that is 500mV

seq_type = []
sti_type = str(dura) + "us_" + str(sti_amp[amp_plot[0]]) + "mV"
print(sti_type)
sti_type = np.repeat(sti_type, repetitions_per_pulse)
seq_type.append(sti_type)
seq_type = np.concatenate(seq_type)

sti_time_plot = []
for i in amp_plot:
    # dura0 [5rep * 7amp] dura1 [..] dura2 [..] dura3 [..] dura4 [..]
    sti_t = sti_time[
        (dura_num * repetitions_per_pulse * amplitudes)
        + repetitions_per_pulse * i : (dura_num)
        * repetitions_per_pulse
        * amplitudes
        + repetitions_per_pulse * i
        + repetitions_per_pulse
    ]
    sti_time_plot.append(sti_t)

stimulation_df["stiTime"] = np.concatenate(sti_time_plot)
stimulation_df["Sequence"] = seq_type

## plot and save the raster plot and PSTH ##
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(hspace=0.3)
all_psth = pop.get_all_psth(
    event=event,
    df=stimulation_df,
    conditions=condition,
    window=window,
    binsize=binsize,
    plot=True,
    colors=["RdYlBu_r"],
)
plt.clim(0, 45)
plt.savefig(
    path
    + filename
    + "_"
    + str(seq_type[1])
    + "_"
    + str(binsize)
    + "ms"
    + "_raster.svg"
)

df_raster = pd.DataFrame(all_psth["data"]["500us_500mV"])
df_raster.to_csv(
    path
    + filename
    + "_"
    + str(seq_type[1])
    + "_"
    + str(binsize)
    + "ms"
    + "_raster.csv",
    index=False,
)

plt.figure(figsize=(10, 5))
pop.plot_population_psth(all_psth=all_psth)
plt.ylim(0, 0.6)
plt.savefig(
    path
    + filename
    + "_"
    + str(seq_type[1])
    + "_"
    + str(binsize)
    + "ms"
    + "_psth.svg"
)
