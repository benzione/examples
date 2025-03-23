#!/usr/bin/env python
"""
Script to run Boruta-SHAP feature selection on a smaller dataset.
This explicitly reports which feature types are being processed.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from tensorflow import keras
from preprocess import preprocess_dataset, train_model
from boruta_shap_selection import run_boruta_shap_selection



def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Boruta-SHAP feature selection')
    parser.add_argument('--dataset', type=str, default='small_dataset.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--model', type=str, default='synthetic_dataset_model.keras',
                        help='Path to the trained model file')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations for Boruta')
    parser.add_argument('--retrain', action='store_true',
                        help='Whether to retrain the model with selected features')
    return parser.parse_args()


def load_and_preprocess_data(dataset_path):
    """
    Load and preprocess the dataset, with detailed reporting on feature types
    """
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Identify original feature types
    numeric_cols = [col for col in df.columns if col.startswith('feature_') and not ('_date' in col or '_cat' in col)]
    date_cols = [col for col in df.columns if '_date' in col]
    cat_cols = [col for col in df.columns if '_cat' in col]
    
    print(f"Original feature counts:")
    print(f"  - Numeric features: {len(numeric_cols)}")
    print(f"  - Date features: {len(date_cols)}")
    print(f"  - Categorical features: {len(cat_cols)}")
    
    # Preprocess the dataset
    df_processed = preprocess_dataset(df)
    print(f"Processed dataset shape: {df_processed.shape}")
    
    # Get processed feature types
    processed_numeric = [col for col in df_processed.columns if 
                     col.startswith('feature_') and not '_' in col]
    
    processed_date = [col for col in df_processed.columns if 
                  any(suffix in col for suffix in ['_year', '_month', '_day', '_dayofweek', '_dayofyear'])]
    
    processed_cat = [col for col in df_processed.columns if 
                 any(f"feature_{i}_cat_" in col for i in range(100))]
    
    print(f"Processed feature counts:")
    print(f"  - Numeric features: {len(processed_numeric)}")
    print(f"  - Date-derived features: {len(processed_date)}")
    print(f"  - One-hot encoded categorical features: {len(processed_cat)}")
    
    # Prepare the target variable
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(df_processed[["category"]])
    y = df_processed["category"].values
    class_names = encoder.categories_[0]
    
    # Remove TIK and category from features
    X = df_processed.drop(columns=["TIK", "category"])
    feature_names = X.columns.tolist()
    
    # Scale numeric features
    numeric_cols = [col for col in X.columns if 
                   not '_' in col or  # Original numeric features
                   col.endswith('_year') or  # Date derived features
                   col.endswith('_month') or
                   col.endswith('_day') or
                   col.endswith('_dayofweek') or
                   col.endswith('_dayofyear')]
    
    # Always create a fresh scaler for the smaller dataset
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, feature_names, y_one_hot, class_names, scaler, encoder


def retrain_with_selected_features(X, y_one_hot, selected_features, original_model_path):
    """
    Retrain a model using only selected features
    """
    from sklearn.model_selection import train_test_split
    
    # Reduce features to selected ones
    X_selected = X[selected_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_one_hot, test_size=0.2, random_state=42
    )
    
    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(X_selected.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(y_one_hot.shape[1], activation="softmax")
    ])
    
    # Compile with same settings as original model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    
    # Train
    print(f"Retraining model with {len(selected_features)} selected features...")
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy with selected features: {score[1]:.4f}")
    
    # Save the model
    model.save("model_selected_features.keras")
    
    return model, history


def main():
    args = parse_arguments()
    
    # Load and preprocess data
    X, y, feature_names, y_one_hot, class_names, scaler, encoder = load_and_preprocess_data(args.dataset)
    
    # Train a simple model if needed
    model = train_model(X, y_one_hot, scaler, encoder)
    
    # Run Boruta with SHAP using the imported function from boruta_shap_selection.py
    print(f"Running Boruta-SHAP feature selection using the model from {args.model}")
    selected_features, importance_df = run_boruta_shap_selection(
        X, y, feature_names, 
        iterations=args.iterations,
        external_model=model
    )
    
    # Print selected features
    print("\nTop 20 selected features:")
    for i, feature in enumerate(selected_features[:20]):
        print(f"{i+1}. {feature}")
    
    # Optionally retrain with selected features
    if args.retrain and selected_features:
        model, history = retrain_with_selected_features(X, y_one_hot, selected_features, args.model)
        print("Model retrained with selected features and saved to model_selected_features.keras")
    
    print("\nFeature selection complete!")

if __name__ == "__main__":
    main() 