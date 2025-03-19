#!/usr/bin/env python
"""
Script to run Boruta-SHAP feature selection on trained model.
This should be run after main.py has completed successfully.
"""

import os
import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Boruta-SHAP feature selection')
    parser.add_argument('--dataset', type=str, default='dataset.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--model', type=str, default='synthetic_dataset_model.keras',
                        help='Path to the trained model file')
    parser.add_argument('--scaler', type=str, default='feature_scaler.pkl',
                        help='Path to the feature scaler file')
    parser.add_argument('--encoder', type=str, default='target_encoder.pkl',
                        help='Path to the target encoder file')
    parser.add_argument('--retrain', action='store_true',
                        help='Whether to retrain the model with selected features')
    parser.add_argument('--manual', action='store_true',
                        help='Use manual implementation of Boruta algorithm')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations for manual Boruta')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Check if required files exist
    required_files = [args.dataset, args.model, args.scaler, args.encoder]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file {file_path} not found.")
            print("Run main.py first to train the model and generate the required files.")
            sys.exit(1)
    
    # Load required packages based on selected method
    if args.manual:
        # Check for manual implementation dependencies
        try:
            import numpy as np
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            import matplotlib.pyplot as plt
            import seaborn as sns
            import shap
        except ImportError as e:
            print(f"Error: Required packages for manual implementation not installed: {e}")
            print("Run: pip install numpy pandas scikit-learn matplotlib seaborn shap")
            sys.exit(1)
            
        # Use manual implementation
        print("Using manual implementation of Boruta algorithm...")
        try:
            from manual_boruta import run_manual_boruta_shap
            from main import preprocess_dataset
            
            # Load and preprocess data
            print("Loading dataset...")
            df = pd.read_csv(args.dataset)
            df_processed = preprocess_dataset(df)
            
            # Load the saved scaler and encoder
            import joblib
            scaler = joblib.load(args.scaler)
            encoder = joblib.load(args.encoder)
            
            # Prepare the target variable
            categories_one_hot = encoder.transform(df_processed[["category"]])
            categories = df_processed["category"].values
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
            
            X[numeric_cols] = scaler.transform(X[numeric_cols])
            
            # Load the trained model
            from tensorflow import keras
            model = keras.models.load_model(args.model)
            
            # Run manual Boruta with SHAP
            final_features, importance_df = run_manual_boruta_shap(
                X, categories, model=model, max_iter=args.iterations
            )
            
            # Print selected features
            print("\nTop 20 selected features:")
            for i, feature in enumerate(final_features[:20]):
                print(f"{i+1}. {feature}")
                
            # Optionally retrain model
            if args.retrain:
                print("\nRetraining model with selected features...")
                from boruta_shap_selection import retrain_with_selected_features
                new_model, history = retrain_with_selected_features(
                    X, categories_one_hot, final_features, args.model
                )
                print("\nFeature selection and model retraining complete!")
            else:
                print("\nFeature selection complete without retraining!")
            
            print("Check the 'plots' directory for feature importance visualizations")
            print("Check 'manual_selected_features.csv' for the list of selected features")
            print("Check 'manual_feature_importance.csv' for detailed feature importance scores")
            
        except Exception as e:
            print(f"Error during manual feature selection: {str(e)}")
            sys.exit(1)
            
    else:
        # Try to use BorutaShap
        try:
            from BorutaShap import BorutaShap
            import shap
            
            from boruta_shap_selection import load_preprocessed_data, boruta_shap_selection, retrain_with_selected_features
            
            # Load preprocessed data
            X, y, feature_names, y_one_hot, class_names = load_preprocessed_data(
                dataset_path=args.dataset,
                scaler_path=args.scaler,
                encoder_path=args.encoder
            )
            
            # Load the trained model
            from tensorflow import keras
            model = keras.models.load_model(args.model)
            
            # Run feature selection
            final_features, importance_df = boruta_shap_selection(
                X, y, y_one_hot, feature_names, class_names, model
            )
            
            # Print selected features
            print("\nTop 20 selected features:")
            for i, feature in enumerate(final_features[:20]):
                print(f"{i+1}. {feature}")
            
            # Optionally retrain model
            if args.retrain:
                print("\nRetraining model with selected features...")
                new_model, history = retrain_with_selected_features(
                    X, y_one_hot, final_features, args.model
                )
                print("\nFeature selection and model retraining complete!")
            else:
                print("\nFeature selection complete without retraining!")
            
            print("Check the 'shap_plots' directory for feature importance visualizations")
            print("Check 'selected_features.csv' for the list of selected features")
            print("Check 'feature_importance.csv' for detailed feature importance scores")
            
        except ImportError as e:
            print(f"Error importing BorutaShap: {e}")
            print("Falling back to manual implementation...")
            print("Try installing BorutaShap directly from GitHub:")
            print("  pip install git+https://github.com/Ekeany/Boruta-Shap.git")
            print("Or use the manual implementation with the --manual flag:")
            print("  python run_feature_selection.py --manual")
            sys.exit(1)
        except Exception as e:
            print(f"Error during feature selection: {str(e)}")
            print("Try using the manual implementation with the --manual flag:")
            print("  python run_feature_selection.py --manual")
            sys.exit(1)

if __name__ == "__main__":
    main() 