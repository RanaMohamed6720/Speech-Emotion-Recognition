# Speech Emotion Recognition (CREMA-D Dataset)

![GitHub](https://img.shields.io/badge/Python-3.8%2B-blue)
![GitHub](https://img.shields.io/badge/Library-Librosa-orange)
![GitHub](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-red)
![GitHub](https://img.shields.io/badge/Dataset-CREMA-green)

---

## Table of Contents

* [Introduction](#introduction)
* [Problem Statement](#problem-statement)
* [Dataset](#dataset)
* [Preprocessing & Feature Extraction](#preprocessing--feature-extraction)
* [Model Architecture](#model-architecture)

  * [2D CNN on Spectrograms](#2d-cnn-on-spectrograms)
  * [1D CNN on Extracted Features](#1d-cnn-on-extracted-features)
* [Training](#training)
* [Evaluation & Results](#evaluation--results)
    * [Confusion matrix](#confusion-matrix)
* [Usage](#usage)
* [Installation](#installation)
* [Authors](#authors)

---

## Introduction

Speech emotion recognition (SER) aims to automatically identify the emotional state of a speaker based on audio signals. This project explores deep learning approaches specifically, convolutional neural networks (CNNs) to classify emotions from speech samples.

## Problem Statement

The objective is to develop models that can distinguish between six basic emotions (anger, disgust, fear, happy, neutral, sad) from audio recordings. We compare two approaches:

1. A **2D CNN** trained on Mel spectrogram representations.
2. A **1D CNN** trained using a manually engineered feature vector composed of time-domain characteristics along with selected frequency-domain features.

## Dataset

We utilize the CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset), which comprises 7,442 audio clips from 91 actors expressing six basic emotions during recitation of twelve sentences ([paperswithcode.com](https://paperswithcode.com/dataset/crema-d?utm_source=chatgpt.com)). Audio clips have a sampling rate of 22,050 Hz.

## Preprocessing & Feature Extraction

* **Audio Loading**: Files are loaded at a sample rate of 22,050 Hz.
* **Signal Processing Parameters**:

  * FFT window size (`n_fft`): 2048 samples
  * Hop length: 512 samples
  * Number of Mel bands: 128

### Spectrogram Features

* **Mel Spectrograms**: generated using the Librosa library and resized to fit the input requirements of a 2D CNN.

### Manually Engineered Feature

#### Time-Domain Features

1. **Zero Crossing Rate (ZCR)**: Rate at which the signal changes sign (positive ↔ negative) per second. Higher ZCR often correlates with high-frequency content (anger, fear).
2. **Root Mean Square Energy (RMS)**: Square root of the mean squared amplitude values per frame, measuring signal energy (loudness).
3. **Frame-wise Mean**: Average amplitude within each frame.
4. **Frame-wise Standard Deviation (STD)**: Variability of amplitude within each frame.

#### Frequency-Domain Features

5. **Spectral Centroid**: Center of mass of the spectrum (average frequency weighted by amplitude). Higher values correspond to brighter or sharper sounds.
6. **Spectral Bandwidth**: Spread of the spectrum around the centroid, indicating frequency dispersion.
7. **Spectral Rolloff**: Frequency below which 85% of the spectral energy resides.

All these features are aggregated per audio clip to form fixed-length input vectors for the 1D CNN.

## Model Architecture

### 2D CNN on Spectrograms

* Convolutional blocks with increasing filter sizes
* Batch normalization and dropout for regularization
* Fully connected layers leading to a softmax output for six classes

### 1D CNN on Extracted Features

* One-dimensional convolutional layers to learn temporal patterns
* Global pooling and dense layers preceding the softmax output

## Training

* **Frameworks**: PyTorch (1D model) and TensorFlow/Keras (2D model)
* **Training Configuration**:

  * Optimizer: AdamW
  * Learning rate: 1e-4
  * Batch size: 64
  * Dropout: 0.2
  * Epochs: 35 (2D), 50 (1D)
  * Early stopping, learning rate reduction on plateau, and model checkpointing

## Evaluation & Results

| Model Type | Train Accuracy | Validation Accuracy | Test Accuracy | F1-Score |
| ---------- | -------------: | ------------------: | ------------: | -------: |
| **2D CNN** |          0.884 |               0.563 |         0.608 |    0.610 |
| **1D CNN** |          0.670 |               0.566 |         0.579 |    0.580 |

The 2D CNN on spectrograms outperforms the 1D CNN on manually engineered features, highlighting the effectiveness of spatial representations of audio for emotion classification.
### Confusion Matrix
![image](https://github.com/user-attachments/assets/3d028198-7633-4f12-9120-9797e6fe162b)

![image](https://github.com/user-attachments/assets/954e701c-c306-49b7-bb20-551e8a16a997)

## Usage

1. Clone the repository.
2. Install dependencies (see below).
3. Place the CREMA-D audio files in the `data/Crema` directory.
4. Launch the Jupyter Notebook:

   ```bash
   jupyter notebook ser-crema-dataset.ipynb
   ```
5. Run all cells to reproduce preprocessing, training, and evaluation.

## Installation

```bash
pip install numpy librosa matplotlib scikit-image scikit-learn pandas torch torchvision torchsummary tensorflow seaborn jupyter
```
## Authors
- Hager Ashraf Mohamed Melook  
- Nouran Ashraf Yousef  
- Rana Mohamed Ali Attia
