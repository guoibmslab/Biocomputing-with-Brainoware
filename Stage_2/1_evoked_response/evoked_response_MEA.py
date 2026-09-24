#!/usr/bin/python
# -------------------------------------------------------------
# Author: Hongwei Cai, Huiyu Chu
# Date:   2025-06-06
# Description: Biocomputing with Brainoware
#              Procedure 2 - Reservoir computing hardware properties
#              1. Measuring evoked responses (with MEA recording)
#              This script describes how to stimulate organoids with different voltage levels and #              phase durations.
# -------------------------------------------------------------

import os
import time
import maxlab as mx
from typing import Optional

from stimulation_example_official import (
    initialize_system,
    load_config,
    connect_stim_units_to_stim_electrodes,
    configure_and_powerup_stim_units,
    create_stim_pulse,
    send_stim_pulses_all_units,
)


def prepare_stim_sequence(
    number_pulses_per_train: int,
    inter_pulse_interval: int,
    phase: int,
    amplitude: int,
    changing_amplitude: Optional[bool] = False,
    max_amplitude: Optional[int] = None,
    amplitude_interval: Optional[int] = None,
    changing_phase: Optional[bool] = False,
    max_phase: Optional[int] = None,
    phase_interval: Optional[int] = None,
) -> mx.Sequence:
    """Prepare a stimulation sequence.

    This is just and example and it only illustrates how sequences
    can be constructed.

    Parameters
    ----------
    number_pulses_per_train : int
        Number of repetitions of one pulse.
    inter_pulse_interval : int
        Number of samples to delay between two consecutive pulses.
    phase : int
        Number of samples before the switch from high-low (and vice versa)
        voltage when creating stimulation pulse.
    amplitude : int
        Amplitude of the pulse (minimal if changing_amplitude is True).
        Unit is millivolt.
    changing_amplitude : Optional[bool]
        Whether the pulse amplitude changes, by default False.
    max_amplitude : Optional[int]
        Maximal amplitude of the pulse if changing_amplitude is True.
        Unit is millivolt.
    amplitude_interval : Optional[int]
        Increment amplitude interval if changing_amplitude is True.
        Unit is millivolt.
    changing_phase : Optional[bool]
        Whether the pulse phase changes, by default False.
    max_phase : Optional[int]
        Maximal phase of the pulse if changing_phase is True.
        Unit is sample (50us per sample).
    phase_interval : Optional[int]
        Increment phase interval if changing_phase is True.
        Unit is sample (50us per sample).
    Returns
    -------
    mx.Sequence
        Sequence object filled with the stimulation sequence.

    """
    seq = mx.Sequence()
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    if changing_amplitude and changing_phase:
        if max_amplitude is None or amplitude_interval is None:
            raise ValueError(
                "Both max_amplitude and amplitude_interval are required for changing_amplitude."
            )
        if max_phase is None or phase_interval is None:
            raise ValueError(
                "Both max_phase and phase_interval are required for changing_phase."
            )
        for cur_amplitude in range(
            amplitude, max_amplitude + amplitude_interval, amplitude_interval
        ):
            for cur_phase in range(
                phase, max_phase + phase_interval, phase_interval
            ):
                for _ in range(number_pulses_per_train):
                    seq = create_stim_pulse(
                        seq, int(cur_amplitude / dac_lsb_mV), cur_phase
                    )
                    seq.append(mx.DelaySamples(inter_pulse_interval))
                seq.append(mx.DelaySamples(inter_pulse_interval))
    elif changing_amplitude:
        if max_amplitude is None or amplitude_interval is None:
            raise ValueError(
                "Both max_amplitude and amplitude_interval are required for changing_amplitude."
            )
        for cur_amplitude in range(
            amplitude, max_amplitude + amplitude_interval, amplitude_interval
        ):
            for _ in range(number_pulses_per_train):
                seq = create_stim_pulse(
                    seq, int(cur_amplitude / dac_lsb_mV), phase
                )
                seq.append(mx.DelaySamples(inter_pulse_interval))
            seq.append(mx.DelaySamples(inter_pulse_interval))
    elif changing_phase:
        if max_phase is None or phase_interval is None:
            raise ValueError(
                "Both max_phase and phase_interval are required for changing_phase."
            )
        for cur_phase in range(
            phase, max_phase + phase_interval, phase_interval
        ):
            for _ in range(number_pulses_per_train):
                seq = create_stim_pulse(
                    seq, int(amplitude / dac_lsb_mV), cur_phase
                )
                seq.append(mx.DelaySamples(inter_pulse_interval))
            seq.append(mx.DelaySamples(inter_pulse_interval))
    else:
        for _ in range(number_pulses_per_train):
            seq = create_stim_pulse(seq, int(amplitude / dac_lsb_mV), phase)
            seq.append(mx.DelaySamples(inter_pulse_interval))
    return seq


if __name__ == "__main__":
    initialize_system()
    config_path = (
        "path/to/your/config.cfg"  # <-- modify based on your configuration
    )
    stim_electrodes = [
        3580,
        4887,
    ]  # <-- modify based on your stimulation electrodes selection
    event_counter = 1  # variable to keep track of the event_id
    array = load_config(config_path)
    stim_units = connect_stim_units_to_stim_electrodes(stim_electrodes, array)
    wells = list(range(1))
    mx.activate(wells)
    array.download(wells)
    time.sleep(
        mx.Timing.waitAfterDownload
    )  # Wait a few seconds to make sure the configuration is downloaded
    mx.offset()
    time.sleep(
        15
    )  # <-- set a larger delay here if more time required for recording in the software
    mx.clear_events()  # Empty event-buffer before adding anything to it

    stim_unit_commands = configure_and_powerup_stim_units(stim_units)

    ## Start recording with maxlab.Saving() ##
    data_directory = "/home/mxwbio/Desktop/evoked_response_data"  # <-- modify the path to your data directory
    os.makedirs(data_directory, exist_ok=True)
    s = mx.Saving()
    s.open_directory(data_directory)
    s.group_delete_all()
    s.group_define(0, "routed")
    print(f"MaxOne: Start recording at {data_directory}")
    time_start = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))
    recording_file_name = f"evoked_response_{time_start}"
    s.start_file(recording_file_name)
    s.start_recording([0])

    # At this stage, we can start the recordings. Note that, if we wish to have events written
    # to the recorded file, it is important to construct our sequence after starting the recording.
    seq = prepare_stim_sequence(
        number_pulses_per_train=10,
        inter_pulse_interval=20000,  # in samples (50us per sample)
        phase=2,  # 2 samples = 100us (4*50us)
        amplitude=100,  # mV
        changing_amplitude=True,
        max_amplitude=700,  # mV
        amplitude_interval=100,  # mV
        changing_phase=True,
        max_phase=10,  # 10 samples = 500us
        phase_interval=2,  # 2 samples = 100us gap
    )
    send_stim_pulses_all_units(seq, number_pulse_trains=1)

    ## Stop recording once all pulses have been delivered ##
    time.sleep(
        389
    )  # 7*5*(10*(1+0.001)+1)=385.35 seconds, add a few seconds of buffer time here
    s.stop_recording()
    s.stop_file()
    print("MaxOne: Recording stopped")
