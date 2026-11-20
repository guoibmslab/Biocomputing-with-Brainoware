# Stage 1 — Spike Sorting & Functional Connectivity

**Biocomputing with Brainoware — Stage 1: Human cortical organoid generation and Brainoware hardware integration**

Authors: Hongwei Cai, Huiyu Chu

This stage takes a raw **MaxOne MEA "Network Scan"** recording of a cortical
organoid (Brainoware), sorts spikes with **Kilosort**, and computes a
**functional connectivity** map between the sorted units using the spike-time
tiling coefficient (STTC).

Two interchangeable implementations are provided:

| Version | Folder | Notebook | Sorter | Runtime |
|---------|--------|----------|--------|---------|
| **Local** | [`1_kilosort_functional_connectivity_local/`](1_kilosort_functional_connectivity_local) | `kilosort.ipynb` | Kilosort2 | Local machine (MATLAB + NVIDIA GPU) |
| **Colab** | [`1_kilosort_functional_connectivity_colab/`](1_kilosort_functional_connectivity_colab) | `Kilosort_colab.ipynb` | Kilosort4 | Google Colab (GPU + High RAM) |

Both notebooks share the same scientific pipeline; they differ mainly in the
sorter version, environment setup.

> **Which one should I run?**  
> - If you don't have a local CUDA GPU + MATLAB and don't want to setup the environment for Kilosort2, use the **Colab** version — it installs all dependencies and downloads the example data automatically.  
> - If you want it to be more challenging and enjoy environment setup, use the **Local** version.


## 1. Common pipeline

```
fc_example_data.raw.h5  (MaxWell MEA Network Scan recording)
        │
        ▼  1. Preprocess: unsigned→signed, band-pass 300–6000 Hz
        ▼  2. Spike sorting  (Kilosort via SpikeInterface)
        ▼  3. Unit locations + firing rates  →  firing-rate map
        ▼  4. STTC functional connectivity  (Numba-accelerated)
        ▼  5. Thresholded adjacency matrix → graph on electrode coordinates
   firing_map.*  +  functional connectivity map  (NetworkX + Plotly)
```

---

## 2. Input data

- **`fc_example_data.raw.h5`** — example MaxOne MEA "Network Scan"
  recording.
  - *Local:* download it from
    [Here](https://www.dropbox.com/scl/fi/yuw74vcwpnr5bnj7juy1w/fc_example_data.raw.h5?rlkey=0ry7a9fmdssw2b451k6k6l7h1&st=dzrdfhrv&dl=0)
    and place it under `Stage_1/1_kilosort_functional_connectivity_local/`
    (alongside `kilosort.ipynb`).
  - *Colab:* downloaded automatically from Dropbox to `/content/`, along with
    `libcompression.so` (the HDF5 plugin required to read MaxOne files).

---

## 3. Version 1 — Local (`kilosort.ipynb`, Kilosort2)

### 3.1 Requirements

| Component | Notes |
|-----------|-------|
| **Kilosort2** | Local [Kilosort 2.0](https://github.com/MouseLand/Kilosort/releases/tag/v2.0) install — requires MATLAB + a CUDA-capable NVIDIA GPU. Download the 2.0 version [here](https://www.dropbox.com/scl/fo/xhr8wnj10d1nh8zu7yc1e/AF4NXMTK4S4XApBOKSiMWjQ?rlkey=gg5ldtm12559nqqem8phftpbh&st=5g4gaaii&dl=0). |
| **MATLAB + toolboxes** | Required MATLAB toolboxes (e.g. Parallel Computing, Signal Processing, Statistics and Machine Learning Toolbox). |
| **C++ compiler** | A MATLAB-compatible C++ compiler. On **Windows**: Visual Studio Community 2017 (MSVC). On **Linux/macOS**: `g++`. |
| **CUDA GPU code** | The GPU (CUDA) commands must be compiled with the C++ compiler before first use. |
| **Python** | 3.8+ |
| Packages | `spikeinterface[full]`, `elephant`, `numpy`, `pandas`, `matplotlib`, `networkx`, `plotly`, `h5py`  (pip install "spikeinterface[full]" elephant numpy pandas matplotlib networkx plotly h5py)|

⚠️Before running, set up the Kilosort2 MATLAB side: install the required 3 MATLAB
toolboxes and a supported C++ compiler, then compile the GPU (CUDA) commands. Refer to the installation instructions in the
[Kilosort2 README](https://github.com/jamesjun/Kilosort2) for more details.

Run the following in the MATLAB Command Window (not a system terminal):

```matlab
cd <your kilosort2 path>/CUDA
mexGPUall
```

> ⚠️ **Warning:** The GPU (CUDA) commands must be compiled successfully in
> MATLAB *before* sorting. This step requires a CUDA toolkit version compatible
> with both your MATLAB release and your C++ compiler — mismatches are the most
> common cause of `mexGPUall` failures. Sorting will not run until compilation
> completes without errors.

Point SpikeInterface at your Kilosort2 install:

```python
ss.Kilosort2Sorter.set_kilosort2_path("D://Kilosort-2.0")
```


### 3.2 Key parameters

**Kilosort2:** `minFR=0.02`, `minfr_goodchannels=0.1`, `detect_threshold=5.5`,
`n_jobs=8`.  
**STTC:** `dt=0.030 s`, `sampling_rate=20000 Hz`.  
**Adjacency / plot:** `thred_c=0.30` (drop weaker edges), `thred_plot=0.35`
(only draw edges above this), `color_map='RdYlGn_r'`.

### 3.3 Troubleshooting

**`UnicodeDecodeError: 'utf-8' codec can't decode byte ...` when running Kilosort2 (Windows)**

On Windows set to Chinese or another language with non-English characters, `run_sorter` can fail with an error like:

```
Error processing .\fc_example_data.raw.h5: 'utf-8' codec can't decode byte 0xb2 in position 5: invalid start byte
```

Your recording file is fine. Before sorting, SpikeInterface (< 0.105) runs a bash check for a compiled Kilosort2. On Windows, `cmd` cannot run it and prints an error in the system language. SpikeInterface then crashes trying to read that message as UTF-8. Linux, macOS and English Windows are usually not affected.

**Fix:** add this line right after `set_kilosort2_path(...)`:

```python
# Skip the bash-based "ks2_compiled" check (crashes on non-UTF-8 Windows).
# This pipeline uses the MATLAB source version of Kilosort2, so the check is not needed.
ss.Kilosort2Sorter.check_compiled = classmethod(lambda cls: False)
```

SpikeInterface will then find Kilosort2 through `master_kilosort.m` in the folder you set. On systems that weren't affected, this line changes nothing.

---

## 4. Version 2 — Colab (`Kilosort_colab.ipynb`, Kilosort4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1_r2YQzQBwS07lJpT5ZoM___KwacIVoAE/view?usp=sharing)

Click the badge above to open the notebook in Google Colab. To keep your own
editable copy, use **File → Save a copy in Drive** once it opens.

### 4.1 Runtime

Run in **Google Colab** with a **GPU runtime + High RAM** (>40 GB, e.g.
A100/H100). Kilosort needs an NVIDIA GPU and large RAM, so Colab is the easiest
way to get a working environment.

### 4.2 Setup (automated in the notebook)

```bash
pip install elephant seaborn
pip install spikeinterface[full,widgets]
pip install kilosort neo "networkx[default]" -U kaleido "plotly>=6.1.1"
```

Cell 3 also downloads `fc_example_data.raw.h5` and `libcompression.so`
(HDF5 plugin) and sets `HDF5_PLUGIN_PATH`.


### 4.3 Key parameters

**Curation:** `snr > 5`, `isi_violations_ratio < 0.3`, `firing_rate > 0.05`.  
**STTC:** `lag_s = 0.030 s`.  
**Adjacency / plot:** `thred_c=0.3`, `thred_plot=0.15`, `color_total=8`,
heatmap range `[-0.15, 0.9]`, `color_map_s='RdYlGn_r'`.


---

## 5. Outputs

Both versions write into their sorter-output folder
(`kilosort2/` locally, `sort_output/` on Colab):

| File | Description |
|------|-------------|
| `sorter_output/` (local) / analyzer Zarr (Colab) | Raw sorter results: spike times, templates, amplitudes, cluster labels, logs |
| `waveforms/` | Waveform extractor used for unit locations (local) |
| `spike_trains*.csv` | Per-unit spike times — columns `unit_id`, `time` (samples) |
| `neuron_coord*.csv` | Per-unit electrode coordinates — `unit_id`, `x`, `y` (µm) [+ firing rate on Colab] |
| `firing_map*.png` / `.svg` | Units at electrode positions, marker size ∝ firing rate |
| `functional connectivity` map | STTC connectivity graph on the electrode layout (+ heatmaps on Colab) |
| `sttc.csv` | Full STTC matrix (local batch loop / Colab) |

---
