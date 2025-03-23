#!/usr/bin/env python
"""
Main script to train a model on the synthetic dataset.
"""

import os
import argparse
from preprocess import preprocess_dataset, train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on the synthetic dataset')
    parser.add_argument('--dataset', type=str, default='dataset.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the model and preprocessing components')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Train the model
    model, X, y, feature_names, class_names = train_model(args.dataset)
    
    print("\nTraining complete!")
    print(f"Model saved to synthetic_dataset_model.keras")
    print(f"Preprocessing components saved to feature_scaler.pkl and target_encoder.pkl")
    
    # Print feature information
    print(f"\nFeature information:")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of classes: {len(class_names)}")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Run feature selection to identify the most important features:")
    print("   python run_feature_selection.py --manual")
    print("\nOr use BorutaShap (if installed):")
    print("   python run_feature_selection.py")
    print("\nWith retraining:")
    print("   python run_feature_selection.py --retrain")

if __name__ == "__main__":
    main()
