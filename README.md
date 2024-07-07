# CNN-for-Predicting-Attachment-Level-of-Bacteria

This repository implements a Convolutional Neural Network (CNN) for classifying and predicting biomaterial attachment levels. It supports both **regression and classification tasks**, enabling precise analysis of biomaterial interactions. The CNN model is designed to handle complex data, providing accurate predictions and classifications to advance research in biomaterial science.

**This Highly scalable framework allows for easy addition of more combinations in the future and can be seamlessly transferred to other projects.**

## Project Structure
- `models/`:
    - `CNN.py`: Implementation of CNN Architecture for Supervised-Learning.
    - `Networks.py`: Implementation of Base Network (parent) definitions and configurations for the CNN architecture.

- `options/`:
    - `base_options.py`: Basic Command-line arguments for the training script.
    - `train_options.py`: Hyperparameter Command-line arguments for the training script.

- `utils/`:
    - `images_utils.py`: Utilities for image handling.
    - `visualizer.py`: This file provides scripts for a TensorBoard visualizer for tracking training progress.
    - `weights_init.py`: This file contains scripts for weight initialization functions for the CNN architecture.

- `train.py`: Script for training the model.
- `call_methods.py`: This file contains scripts for dynamically creating models, networks, datasets, and data loaders based on provided names and options.

## Requirements
To run the code, you need the following:

Python 3.8 or above
PyTorch 1.7 or above
torchvision
tqdm
matplotlib
TensorboardX 2.7.0
Install the necessary packages using pip:

