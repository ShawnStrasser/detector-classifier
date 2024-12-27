# detector-classifier
Predicting traffic signal detector phase and function from hi-res event logs, to aid in configuring detectors for ATSPMs. Work in progress.

## Kaggle Competition

This repository contains the code for the [Traffic Signal Detector Classifier](https://www.kaggle.com/c/traffic-signal-detector-classifier) competition on Kaggle. The datasets can be downloaded from the competition page.

## Model Architecture

The detector classifier uses a deep learning architecture to predict both the phase and function of traffic signal detectors based on their activation patterns and signal phase timing data.

### Input
- Sequence length: 300 timesteps
- Features per timestep: 10
  - 8 channels for phase states
  - 1 channel for detector state
  - 1 channel for time delta between events

### Architecture Layers

Input Layer (300 timesteps × 10 features)<br>
↓<br>
Masking Layer (for handling padded sequences)<br>
↓<br>
Bidirectional LSTM Layer 1<br>

64 units per direction
return sequences enabled
total 128 features per timestep output
<br>↓<br>
Dropout Layer (0.2)
<br>↓<br>
Bidirectional LSTM Layer 2
32 units per direction
total 64 features output
<br>↓<br>
Dropout Layer (0.2)
<br>↓<br>
Dense Layer
64 units
ReLU activation
<br>↓<br>
Output Branches:
├→ Phase Output
│ - Dense layer (8 units)
│ - Softmax activation
│ - Predicts 8 possible phases
│
└→ Function Output
- Dense layer (3 units)
- Softmax activation
- Predicts 3 possible functions
(Advance, Presence, Count)

### Loss Function
Combined weighted loss:
total_loss = α * phase_loss + β * function_loss

where:
- phase_loss: Categorical crossentropy for phase prediction
- function_loss: Categorical crossentropy for function prediction
- α, β: Configurable weight parameters (default: α = β = 0.5)