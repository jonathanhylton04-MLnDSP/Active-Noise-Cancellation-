# Active Noise Cancellation: Offline, Adaptive & Deep Learning Methods

> A MATLAB simulation comparing Wiener ANC, FxLMS, NFxLMS, and Deep ANC Lite for active noise cancellation using real acoustic path data and real-world noise recordings.

---

## Table of Contents

- [Overview](#overview)
- [Methods](#methods)
- [Results](#results)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Output](#output)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [Author](#author)

---

## Overview

This project implements and benchmarks four active noise cancellation (ANC) methods in MATLAB. The noisy signal is modeled as:

```
noisy signal = clean music + primary noise
```

A reference noise signal is fed to the ANC controller, which generates a canceling signal. That signal passes through a secondary path before being subtracted at the error microphone.

**Simulation Parameters**

| Parameter | Value |
|---|---|
| Sampling Rate | 16 kHz |
| Duration | 20 seconds |
| Input SNR | 0 dB |
| Primary Path Taps | 256 |
| Secondary Path Taps | 512 |

---

## Methods

### Wiener ANC
An offline Wiener filter solution that estimates a controller from the filtered reference noise to the primary noise. Because it has access to the target primary noise during filter design, it serves as an ideal upper-bound reference rather than a practical real-time method.

### FxLMS
A standard adaptive ANC method. The filter is updated using the filtered reference signal to account for the secondary path between the controller output and the error microphone. Sensitive to step size selection.

### NFxLMS
The normalized variant of FxLMS. The update step is divided by the filtered reference signal power, producing more stable convergence than standard FxLMS.

### Deep ANC Lite
A simplified deep learning approach inspired by the Deep ANC research papers. A small regression neural network is trained to map:

```
STFT(reference noise) → STFT(canceling signal)
```

The Wiener canceling signal is used as the training target. The model predicts the real and imaginary parts of the canceling spectrogram, then reconstructs the waveform via inverse STFT.

> **Note:** This is not a full reproduction of the Deep ANC paper. It is a simplified MATLAB proof-of-concept following the same core idea, without the full convolutional recurrent network (CRN).

---

## Results

### SNR Improvement

| Method | Avg. SNR Improvement |
|---|---|
| Wiener ANC | **44.3 dB** |
| NFxLMS | 4.0 dB |
| Deep ANC Lite | 2.1 dB |
| FxLMS | 1.3 dB |

### STOI Scores

| Method | Avg. STOI |
|---|---|
| Wiener ANC | **1.000** |
| NFxLMS | 0.680 |
| FxLMS | 0.666 |
| Deep ANC Lite | 0.652 |
| Noisy Input | 0.587 |

### Key Takeaways

- **Wiener ANC** achieves the highest performance as an offline optimal method — it is best understood as a theoretical ceiling.
- **NFxLMS** outperforms standard FxLMS due to the stabilizing effect of normalization.
- **Deep ANC Lite** successfully demonstrates the concept but, as expected, falls short of the Wiener solution given the simplified network architecture.

---

## Datasets

### MS-SNSD (Noise Files)
Noise recordings come from the [Microsoft Scalable Noisy Speech Dataset](https://github.com/microsoft/MS-SNSD). Files are drawn from the `noise_test` folder across five noise classes:

- AirConditioner
- Babble
- Neighbor
- ShuttingDoor
- AirportAnnouncements

### PANDAR (Acoustic Path Data)
The secondary path impulse response comes from the [PANDAR ANC Path Database](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/paths-for-active-noise-cancellation-development-and-research/), using measured paths from the Bose QC20 in-ear headphone in an acoustic booth environment.

---

## Project Structure

```
project_folder/
│
├── ECE6095Project.m # Main MATLAB script
├── Its Over.mp3 # Music source file
│
├── MS-SNSD/ # Download separately — see How to Run
│   └── noise_test/
│       ├── AirConditioner_1.wav
│       ├── Babble_1.wav
│       ├── Neighbor_1.wav
│       ├── ShuttingDoor_1.wav
│       └── AirportAnnouncements_1.wav
│
├── PANDAR_database_1.0/ # Download separately — see How to Run
│   └── BoseQC20/
│       └── acoustic_booth/
│           └── persons/
│               └── PANDAR_TF_001_person_BoseQC20.ita
│
├── toolbox/ # Download separately — see How to Run
│
├── project_figures/ # Auto-created by script
└── project_results/ # Auto-created by script
```

---

## Requirements

Built and tested in **MATLAB**. The following toolboxes are recommended:

- Signal Processing Toolbox
- Audio Toolbox
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox

---

## How to Run

### 1. Clone this repository
```bash
git clone https://github.com/jonathanhylton04-MLnDSP/Active-Noise-Cancellation-.git
```

### 2. Download MS-SNSD (Noise Dataset)
1. Go to https://github.com/microsoft/MS-SNSD
2. Clone or download the repo

### 3. Download PANDAR Database (Acoustic Paths)
1. Go to https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/paths-for-active-noise-cancellation-development-and-research/
2. Download the PANDAR database

### 4. Add your music file
Place your music file (`Its Over.mp3`) in the project root folder.

### 5. Run in MATLAB
Open MATLAB, set the working directory to the project folder, and run:
```matlab
ECE6095Project
```

---

## Output

The script prints SNR improvement and STOI scores for each method to the console, and saves the following to disk:

**Figures** → `project_figures/`

**Result tables** → `project_results/`
- `snr_results.csv`
- `stoi_results.csv`
- `summary_results.csv`

---

## Limitations & Future Work

### Current Limitations
- Wiener ANC uses offline target information — not suitable for real-time deployment.
- Deep ANC Lite uses a small, simplified network rather than a full CRN architecture.
- Only five noise classes were tested.
- The primary path was simulated rather than measured.
- Training and testing used similar (rather than fully held-out) clips.

### Possible Improvements
- Use more noise files across more diverse conditions for training and testing.
- Split training and testing by file to avoid data leakage.
- Upgrade to a full CRN- or LSTM-based network for Deep ANC Lite.
- Train across multiple measured acoustic paths.
- Implement real-time streaming ANC.
- Incorporate measured primary paths from PANDAR where available.
- Evaluate on speech signals for more meaningful STOI interpretation.

---

## References

1. S. M. Kuo and D. R. Morgan, "Active noise control: A tutorial review," *Proceedings of the IEEE*, vol. 87, no. 6, pp. 943–973, Jun. 1999.
2. H. Zhang and D. Wang, "A deep learning approach to active noise control," in *Proc. Interspeech 2020*, Shanghai, China, Oct. 2020, pp. 1141–1145.
3. H. Zhang and D. Wang, "Deep ANC: A deep learning approach to active noise control," *Neural Networks*, vol. 141, pp. 1–10, Sep. 2021.
4. C. K. A. Reddy, E. Beyrami, J. Pool, R. Cutler, S. Srinivasan, and J. Gehrke, "A scalable noisy speech dataset and online subjective test framework," in *Proc. Interspeech 2019*, Graz, Austria, Sep. 2019, pp. 1816–1820.
5. Microsoft, "Microsoft Scalable Noisy Speech Dataset," GitHub repository, 2021. [Online]. Available: https://github.com/microsoft/MS-SNSD. Accessed: May 1, 2026.
6. Institute of Communication Systems, RWTH Aachen University, "Paths for Active Noise Cancellation Development and Research," 2019. [Online]. Available: https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/paths-for-active-noise-cancellation-development-and-research/. Accessed: May 1, 2026.
7. S. Liebich, J. Fabry, P. Jax, and P. Vary, "Acoustic path database for ANC in-ear headphone development," in *Proc. 23rd International Congress on Acoustics*, Aachen, Germany, Sep. 2019, pp. 4326–4333.
8. C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, "An algorithm for intelligibility prediction of time frequency weighted noisy speech," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 19, no. 7, pp. 2125–2136, Sep. 2011.
9. MathWorks, "stoi, Short time objective intelligibility measure," 2026. [Online]. Available: https://www.mathworks.com/help/audio/ref/stoi.html. Accessed: May 1, 2026.
10. M. Escabi, *Audio Signal Processing and Machine Learning*, Presentation 5, Wiener Filter and Audio Calibration. University of Connecticut, 2026.
11. J. Hylton, *Wiener Filter, ANC, and Deep ANC Notes*. University of Connecticut, 2026.


---

## Author

**Jonathan Hylton**  
**Electrical Engineering**  
**Digital Signal Processing and Machine Learning for Speech, Audio & Music**  
**University of Connecticut, Storrs, CT**  
