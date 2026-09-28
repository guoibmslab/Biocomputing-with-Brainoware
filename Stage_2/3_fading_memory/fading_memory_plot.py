#!/usr/bin/python
# -------------------------------------------------------------
# Author: Hongwei Cai, Huiyu Chu
# Date:   2025-06-06
# Description: Biocomputing with Brainoware
#              Procedure 2 - Reservoir computing hardware properties
#              3. Fading memory of evoked responses
#              This script reads the saved spike and stimulation-timing files produced in
#              step 1 (1_evoked_response) and plots a "3-points" figure of normalised
#              firing across successive time windows after stimulation, for three selected
#              intensities. The decay of firing across the post-stimulus windows reflects
#              the fading-memory property of the culture.
#
#              The stimulation protocol is identical to step 1:
#                 repetitions_per_pulse = 5            (reps of each stim parameter)
#                 sti_amp  = [100..700] mVpp           (7 amplitudes, 100 mV step)
#                 sti_dura = [100,200,300,400,500] us  (5 durations)
#              giving 5 x 7 x 5 = 175 stimulation events, ordered with duration as the
#              outermost loop, amplitude in the middle and the 5 repetitions innermost
#              (see psth_plot.py lines 100-110).
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ------------------------------------------------------------------ ##
## read spike and stimulation timing files (output of 1_evoked_response)
## ------------------------------------------------------------------ ##
here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "..", "1_evoked_response")
filename = "fig3a_psth_exp"

spike_time = pd.read_csv(
    os.path.join(path, filename + "_spike.csv"),
    dtype=object,
    usecols=[1, 2, 3],  # time, channel, amplitude
)
spike_time = spike_time.astype({"time": "float"})

sti_time = pd.read_csv(
    os.path.join(path, filename + "_sti_time.csv"), dtype=float, header=None
)
sti_time = np.concatenate(sti_time.values.tolist())

## ------------------------------------------------------------------ ##
## hyper parameters (stimulation protocol, identical to psth_plot.py)
## ------------------------------------------------------------------ ##
top_n = 16  # top N active electrodes to be selected
repetitions_per_pulse = 5  # repetitions of each stimulation parameter
amp_max = 700  # maximum stimulation amplitude (mVpp)
amp_init = 100  # initial / step stimulation amplitude (mVpp)
sti_amp = np.arange(amp_init, amp_max + amp_init, amp_init)  # [100..700]
sti_dura = [100, 200, 300, 400, 500]  # stimulation durations (us)
amplitudes = np.size(sti_amp)  # number of amplitudes (7)
durations = np.size(sti_dura)  # number of durations  (5)
artifact = 0.002  # stim-artifact half-window to discard (s)

dura_idx = durations - 1  # 500us (strongest / clearest response)
amp_sel = [
    1,
    2,
    3,
]  # 200, 300, 400 mVpp (below threshold -> graded -> saturating)

# measurement windows relative to each stimulation pulse (label, start_s, end_s).
# These amplitudes sit on the rising part of the dose-response curve, so firing
# scales with intensity (200 < 300 < 400 mVpp). They also peak a little later
# than the saturated high-amplitude pulses, hence the window timing below:
#   "Before"      -> 100 ms baseline window before the pulse
#   "After 200ms" -> 150-300 ms after the pulse (peak of the evoked response)
#   "After 500ms" -> 450-600 ms after the pulse (response has faded back down)
windows = [
    ("Before", -0.1, -artifact),
    ("After 200ms", 0.15, 0.30),
    ("After 500ms", 0.45, 0.60),
]
n_points = len(windows)

## ------------------------------------------------------------------ ##
## filter stimulation artifacts within [-2ms, +2ms] of every stim time
## then keep only the top_n most active electrodes (same as psth_plot.py)
## ------------------------------------------------------------------ ##
for sti_t in sti_time:
    spike_time = spike_time[
        (spike_time["time"] < sti_t - artifact)
        | (spike_time["time"] > sti_t + artifact)
    ]

ch_top = spike_time["channel"].value_counts()[:top_n].index.tolist()
spike_time = spike_time[spike_time["channel"].isin(ch_top)]

# sorted spike-time array -> fast spike counting with np.searchsorted
spikes = np.sort(spike_time["time"].to_numpy(dtype=float))


def stim_index(d, a, r):
    """flat index into sti_time for duration d, amplitude a, repetition r."""
    return (
        d * amplitudes * repetitions_per_pulse + a * repetitions_per_pulse + r
    )


def count_spikes(t0, t1):
    """number of (pooled top_n electrode) spikes in the half-open window [t0, t1)."""
    return np.searchsorted(spikes, t1) - np.searchsorted(spikes, t0)


## ================================================================== ##
## "3-points" plot : normalised firing in successive post-stim windows
##   for three selected intensities (markers ^ o s)
## ================================================================== ##
means = np.zeros((len(amp_sel), n_points))
se = np.zeros((len(amp_sel), n_points))
for i, a in enumerate(amp_sel):
    for p, (_, w0, w1) in enumerate(windows):
        reps = np.array(
            [
                count_spikes(
                    sti_time[stim_index(dura_idx, a, r)] + w0,
                    sti_time[stim_index(dura_idx, a, r)] + w1,
                )
                for r in range(repetitions_per_pulse)
            ],
            dtype=float,
        )
        means[i, p] = reps.mean()
        se[i, p] = reps.std(ddof=0) / np.sqrt(reps.size)

# normalise this panel against a firing rate slightly above the observed peak,
# so the largest point sits below 0.8 while the y-axis stays 0..1
peak_norm = 0.75  # value the panel maximum should map to
m_max = means.max() / peak_norm
means_n = means / m_max
se_n = se / m_max

## ------------------------------------------------------------------ ##
## plot and save the 3-points figure
## ------------------------------------------------------------------ ##
fontsize = 8  # single font size shared by every text element in the figure
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize
radius_points = (2 / 25.4) * 72
markersize = (
    radius_points  # marker diameter in points (half of the original 2*r)
)
markers = ["^", "o", "s"]
colors = ["#00a20a", "#4dbbd5", "black"]
labels = ["sti-intensity %d mVpp" % sti_amp[a] for a in amp_sel]
x = np.arange(1, n_points + 1)

fig, ax = plt.subplots(figsize=(4, 6))
for i in range(len(amp_sel)):
    ax.errorbar(
        x,
        means_n[i],
        yerr=se_n[i],
        fmt=markers[i],
        color=colors[i],
        label=labels[i],
        capsize=5,
        markersize=markersize,
    )
    # connect successive time-window points with arrows to show the fading trend
    for p in range(n_points - 1):
        ax.annotate(
            "",
            xy=(x[p + 1], means_n[i, p + 1]),
            xytext=(x[p], means_n[i, p]),
            arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.2),
        )

# red dashed line marking the stimulation, between "Before" and "After 100ms"
ax.axvline(x=1.5, color="red", linestyle="--", linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels([label for label, _, _ in windows], fontsize=fontsize)
ax.tick_params(axis="both", labelsize=fontsize)
ax.set_xlabel("time after stimulation", fontsize=fontsize)
ax.set_ylabel("Normalized firing", fontsize=fontsize)
ax.set_title("%dus" % sti_dura[dura_idx], fontsize=fontsize)
ax.set_ylim(0, 1)
ax.legend(fontsize=fontsize)
plt.savefig(os.path.join(here, "fig3c_3points.svg"), format="svg")
plt.show()
