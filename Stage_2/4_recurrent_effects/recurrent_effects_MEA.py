#!/usr/bin/python
# -------------------------------------------------------------
# Author: Hongwei Cai, Huiyu Chu
# Date:   2025-06-06
# Description: Biocomputing with Brainoware
#              Procedure 2 - Reservoir computing hardware properties
#              4. Recurrent effects (with MEA recording)
#              This script describes recurrent effects of organoid
#              reservoir computing with single or multiple pulses.
#              This is the recording-enabled version of recurrent_effects_MEA.py: it wraps
#              the stimulation in a maxlab.Saving() recording so the responses are saved.
# -------------------------------------------------------------

import os
import time
import maxlab as mx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stimulation_example_official import (
    initialize_system,
    configure_array,
    connect_stim_units_to_stim_electrodes,
    configure_and_powerup_stim_units,
    create_stim_pulse,
    send_stim_pulses_all_units,
)


def prepare_stim_sequence(
    number_pulses_per_train: int,
    inter_pulse_interval: int,
    inter_train_interval: int,
    phase: int,
    amplitude: int,
) -> mx.Sequence:
    seq = mx.Sequence()
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    repeating_number = 1
    for _ in range(number_pulses_per_train):
        for rep in range(repeating_number):
            seq = create_stim_pulse(seq, int(amplitude / dac_lsb_mV), phase)
            seq.append(mx.DelaySamples(inter_pulse_interval))
        repeating_number += 1
        seq.append(mx.DelaySamples(inter_train_interval))
        seq.append(mx.DelaySamples(inter_train_interval))
    return seq


if __name__ == "__main__":
    initialize_system()
    electrodes = [
        4885,
        4666,
        4886,
        4022,
        5327,
        5328,
        5106,
        5326,
        3138,
        3140,
        2919,
        5105,
        4667,
        4448,
        5109,
        4669,
        4665,
        3798,
        4021,
        3141,
        4668,
        4240,
        3363,
        3803,
        3580,
        3801,
        2921,
        3799,
        4239,
        3359,
        3142,
        3797,
        3361,
    ]
    stim_electrodes = [3580, 4887]
    event_counter = 1  # variable to keep track of the event_id
    array = configure_array(electrodes, stim_electrodes)
    stim_units = connect_stim_units_to_stim_electrodes(stim_electrodes, array)
    wells = list(range(1))
    mx.activate(wells)
    array.download(wells)
    # Wait a few seconds to make sure the configuration is downloaded
    time.sleep(mx.Timing.waitAfterDownload)
    mx.offset()
    # Wait a few more seconds to make sure the offset compensation is done
    time.sleep(15)
    mx.clear_events()  # Empty event-buffer before adding anything to it

    stim_unit_commands = configure_and_powerup_stim_units(stim_units)

    ## Start recording with maxlab.Saving() ##
    # Open a Saving object, point it at the data directory, and define the
    # "routed" recording group. Recording must be started BEFORE the sequence
    # is constructed/sent so that the stimulation events are written into the
    # recorded file (see the note below).
    data_directory = "/home/mxwbio/Desktop/recurrent_effects_data"  # <-- modify the path to your data directory
    os.makedirs(data_directory, exist_ok=True)
    s = mx.Saving()
    s.open_directory(data_directory)
    s.group_delete_all()
    s.group_define(0, "routed")
    print(f"MaxOne: Start recording at {data_directory}")
    time_start = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))
    recording_file_name = f"recurrent_effects_{time_start}"
    s.start_file(recording_file_name)
    s.start_recording([0])

    # At this stage, we can start the recordings. Note that, if we wish to have events written
    # to the recorded file, it is important to construct our sequence after starting the recording.
    seq = prepare_stim_sequence(
        number_pulses_per_train=4,
        inter_pulse_interval=8000,  # samples
        inter_train_interval=20000,  # samples
        phase=2,  # 2 samples = 100us (4*50us)
        amplitude=100,  # mV
    )
    send_stim_pulses_all_units(seq, number_pulse_trains=1)

    ## Stop recording once all pulses have been delivered ##
    time.sleep(
        15
    )  # 2*4 + 10*(0.4+0.0002)= 12.002 seconds, add a few seconds of buffer time here
    s.stop_recording()
    s.stop_file()
    print("MaxOne: Recording stopped")
