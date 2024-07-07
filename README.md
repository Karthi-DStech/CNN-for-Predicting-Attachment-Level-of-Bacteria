# CNN


- `models/`:
    - `CNN.py`:
    - `Networks.py`:

- `options/`:
    - `base_options.py`: Basic Command-line arguments for the training script.
    - `train_options.py`: Hyperparameter Command-line arguments for the training script.

- `utils/`:
    - `images_utils.py`: Utilities for image handling.

    - `visualizer.py`: This file provides scripts for a TensorBoard visualizer for tracking training progress.
    - `utils.py`:  This file contains scripts for various utility functions used in the GAN project.
    - `weights_init.py`: This file contains scripts for weight initialization functions for the GAN architecture.

- `train.py`: Script for training the model without TensorBoard logging.
- `call_methods.py`: This file contains scripts for dynamically creating models, networks, datasets, and data loaders based on provided names and options.
