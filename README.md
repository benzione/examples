# Feature Selection with Boruta-SHAP

This project implements a machine learning pipeline with feature selection using a combination of Boruta algorithm and SHAP (SHapley Additive exPlanations) values.

## Overview

The main workflow consists of:

1. Training a neural network model on a dataset (via `main.py`)
2. Performing feature selection using Boruta and SHAP values (via `boruta_shap_selection.py`)
3. Optionally retraining the model with only the selected features

## Installation

Install the required packages:

```bash
pip install -r requirments.txt
```

Note: If you have issues installing `BorutaShap`, you can install it directly from GitHub:

```bash
pip install git+https://github.com/Ekeany/Boruta-Shap.git
```

Alternatively, you can use our manual implementation which doesn't require external Boruta packages:

```bash
python run_feature_selection.py --manual
```

## Usage

### Step 1: Train the initial model

First, run the main script to train the model on the full dataset:

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Train a neural network model
- Save the model, scaler, and encoder

### Step 2: Run Boruta-SHAP feature selection

After training the model, run the feature selection:

```bash
python run_feature_selection.py
```

Options:
- `--dataset`: Path to the dataset CSV file (default: dataset.csv)
- `--model`: Path to the trained model file (default: synthetic_dataset_model.keras)
- `--scaler`: Path to the feature scaler file (default: feature_scaler.pkl)
- `--encoder`: Path to the target encoder file (default: target_encoder.pkl)
- `--retrain`: Retrain the model with selected features (flag, default: false)
- `--manual`: Use manual implementation of Boruta (flag, default: false)
- `--iterations`: Number of iterations for manual Boruta (default: 100)

Example with retraining:
```bash
python run_feature_selection.py --retrain
```

Example with manual implementation:
```bash
python run_feature_selection.py --manual --iterations 50 --retrain
```

### Step 3: Examine the results

After running feature selection, check the following outputs:

For the standard implementation:
- `selected_features.csv`: List of selected features
- `feature_importance.csv`: Detailed feature importance scores
- `shap_plots/`: Directory containing visualizations:
  - `shap_summary.png`: SHAP summary plot
  - `shap_importance.png`: Feature importance from SHAP values
  - `top_features_shap.png`: Top 20 features by SHAP importance
  - `combined_top_features.png`: Combined top features from Boruta and SHAP

For the manual implementation:
- `manual_selected_features.csv`: List of selected features
- `manual_feature_importance.csv`: Detailed feature importance scores
- `plots/`: Directory containing visualizations:
  - `manual_boruta_importances.png`: Feature importances from manual Boruta
  - `manual_combined_importances.png`: Combined importances from Boruta and SHAP

If retraining was performed, also check:
- `synthetic_dataset_model_selected_features.keras`: Model trained with selected features
- `training_history_selected_features.png`: Training history plot

## Feature Selection Methods

### Boruta

Boruta is an all-relevant feature selection algorithm that works by comparing the importance of real features with importance achievable at random (using shadow features). Features that consistently perform better than random are kept.

We use `BorutaShap`, which is an implementation that directly combines the Boruta algorithm with SHAP values for feature importance ranking.

If the `BorutaShap` package is not available, we also provide a manual implementation in `manual_boruta.py` that follows the same principles but doesn't require external Boruta packages.

### SHAP Values

SHAP (SHapley Additive exPlanations) values explain the contribution of each feature to the prediction. They provide a unified measure of feature importance based on game theory principles.

### Combined Approach

Our implementation:
1. Uses `BorutaShap` to identify statistically significant features
2. Also calculates SHAP values from the deep learning model to rank features by their contribution to the model's predictions
3. Combines both sets of features for a robust selection
4. Provides visualizations to understand feature importance

## File Structure

- `main.py`: Main script for training the neural network
- `boruta_shap_selection.py`: Implementation of Boruta-SHAP feature selection
- `run_feature_selection.py`: Script to run the feature selection process
- `manual_boruta.py`: Manual implementation of Boruta algorithm
- `requirments.txt`: Required packages