# CNN-for-Predicting-Attachment-Level-of-Bacteria

This repository implements a Convolutional Neural Network (CNN) for classifying and predicting biomaterial attachment levels. It supports both **regression and classification tasks**, enabling precise analysis of biomaterial interactions. The CNN model is designed to handle complex data, providing accurate predictions and classifications to advance research in biomaterial science.

# Project Structure
- `models/`:
    - `CNN.py`: Implementation of CNN Architecture for Supervised-Learning.
    - `Networks.py`: Implementation of Base Network (parent) definitions and configurations for the CNN architecture.

- `options/`:
    - `base_options.py`: Basic Command-line arguments for the training script.
    - `train_options.py`: Hyperparameter Command-line arguments for the training script.

- `utils/`:
    - `images_utils.py`: Utilities for image handling.

    - `visualizer.py`: This file provides scripts for a TensorBoard visualizer for tracking training progress.
    - `utils.py`:  This file contains scripts for various utility functions used in the GAN project.
    - `weights_init.py`: This file contains scripts for weight initialization functions for the GAN architecture.

- `train.py`: Script for training the model.
- `call_methods.py`: This file contains scripts for dynamically creating models, networks, datasets, and data loaders based on provided names and options.
