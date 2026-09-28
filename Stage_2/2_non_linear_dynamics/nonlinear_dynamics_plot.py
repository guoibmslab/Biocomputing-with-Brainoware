#!/usr/bin/python
# -------------------------------------------------------------
# Author: Hongwei Cai, Huiyu Chu
# Date:   2025-06-06
# Description: Biocomputing with Brainoware
#              Procedure 2 - Reservoir computing hardware properties
#              2. Non-linear dynamics of evoked responses
#              This script reads the saved spike and stimulation-timing files produced in
#              step 1 (1_evoked_response) and plots a family of dose-response curves
#              (normalised firing vs stimulation intensity), one curve per stimulation
#              duration, overlaid with a fitted sigmoid showing the saturating, non-linear
#              response of the culture.
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
from scipy.optimize import curve_fit

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

resp_window = (
    0.5  # total post-stim window used for the dose-response curve (s)
)

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
## dose-response curves (normalised firing vs intensity)
##   one curve per stimulation duration, with a fitted sigmoid overlay
## ================================================================== ##
# mean / SE evoked spike count (over the 5 repetitions) for every (duration, amplitude)
resp_mean = np.zeros((durations, amplitudes))
resp_se = np.zeros((durations, amplitudes))
for d in range(durations):
    for a in range(amplitudes):
        reps = np.array(
            [
                count_spikes(
                    sti_time[stim_index(d, a, r)] + artifact,
                    sti_time[stim_index(d, a, r)] + resp_window,
                )
                for r in range(repetitions_per_pulse)
            ],
            dtype=float,
        )
        resp_mean[d, a] = reps.mean()
        resp_se[d, a] = reps.std(ddof=0) / np.sqrt(reps.size)

# normalise the whole family so the curves stay comparable; scale by a slightly
# larger firing rate than the observed peak so the maximum sits around 0.8
peak_norm = 0.8  # value the global maximum should map to
g_max = resp_mean.max() / peak_norm
resp_mean_n = resp_mean / g_max
resp_se_n = resp_se / g_max


def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


# fit the sigmoid to the mean dose-response of the most responsive durations
# (200us + 300us); the other pulses pull the plateau away from the strongest
# responses, so the curve here plateaus around ~0.63
x_idx = np.arange(amplitudes)
mean_curve = resp_mean_n[[1, 2]].mean(axis=0)
initial_guess = [mean_curve.max(), np.median(x_idx), 1.0, mean_curve.min()]
popt, _ = curve_fit(sigmoid, x_idx, mean_curve, p0=initial_guess, maxfev=10000)
x_fit = np.linspace(x_idx.min(), x_idx.max(), 100)
y_fit = sigmoid(x_fit, *popt)

## ------------------------------------------------------------------ ##
## plot and save the dose-response figure
## ------------------------------------------------------------------ ##
fontsize = 8  # single font size shared by every text element in the figure
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize
radius_points = (1.876 / 25.4) * 72
markersize = (
    radius_points  # marker diameter in points (half of the original 2*r)
)
labels = ["%dus" % d for d in sti_dura]
# 100us, 200us, 300us, 400us, 500us  (200us -> grey, 400us -> orange so the
# two mid curves no longer sit on top of each other)
colors = ["#c5c1a4", "#808080", "#dc0f0f", "#f69a47", "#3c5488"]

plt.figure(figsize=(8, 6))
for i in range(durations):
    plt.errorbar(
        x_idx,
        resp_mean_n[i],
        yerr=resp_se_n[i],
        label=labels[i],
        fmt="o",
        color=colors[i],
        capsize=5,
        markersize=markersize,
    )
plt.plot(x_fit, y_fit, label="Fitted Sigmoid", color="red", linestyle="--")
plt.xticks(x_idx, sti_amp, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel("Intensity (mVpp)", fontsize=fontsize)
plt.ylabel("Normalized Firing", fontsize=fontsize)
plt.title("Evoked response vs stimulation intensity", fontsize=fontsize)
plt.ylim(0, 1)
plt.legend(fontsize=fontsize)
plt.grid(False)
plt.savefig(os.path.join(here, "fig3b_response_curve.svg"), format="svg")
plt.show()
