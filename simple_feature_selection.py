#!/usr/bin/env python
"""
Simplified feature selection that handles all feature types
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from preprocess import preprocess_dataset

def load_and_preprocess_data(dataset_path):
    """
    Load and preprocess the dataset, reporting on feature types
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
    
    # Create a fresh scaler
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, feature_names, y_one_hot, class_names

def simple_feature_selection(X, y, feature_names, n_estimators=100):
    """
    Simple feature selection using RandomForest importance
    """
    # Convert target to numeric
    le = LabelEncoder()
    y_numeric = le.fit_transform(y)
    
    # Train a RandomForest
    print("Training RandomForest for feature selection...")
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    rf.fit(X, y_numeric)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create DataFrame with importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create plot directory
    os.makedirs('plots', exist_ok=True)
    
    # Define threshold for feature selection (e.g., top 30% of features)
    threshold = np.percentile(importances, 70)
    selected_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()
    
    print(f"Selected {len(selected_features)} features (top 30%)")
    
    # Categorize selected features by type
    numeric_selected = [f for f in selected_features if 
                      f.startswith('feature_') and 
                      not any(suffix in f for suffix in ['_cat_', '_year', '_month', '_day', '_dayofweek', '_dayofyear'])]
    
    date_selected = [f for f in selected_features if 
                  any(suffix in f for suffix in ['_year', '_month', '_day', '_dayofweek', '_dayofyear'])]
    
    cat_selected = [f for f in selected_features if 
                any(f"feature_{i}_cat_" in f for i in range(100))]
    
    print("\nSelected features by type:")
    print(f"  - Numeric features: {len(numeric_selected)}")
    print(f"  - Date-derived features: {len(date_selected)}")
    print(f"  - One-hot encoded categorical features: {len(cat_selected)}")
    
    # Create a more categorical representation of feature types for visualization
    importance_df['Type'] = importance_df['Feature'].apply(lambda f: 
        'Numeric' if f.startswith('feature_') and not any(suffix in f for suffix in ['_cat_', '_year', '_month', '_day', '_dayofweek', '_dayofyear']) else
        'Date' if any(suffix in f for suffix in ['_year', '_month', '_day', '_dayofweek', '_dayofyear']) else
        'Categorical'
    )
    
    # Save selected features and importance
    pd.DataFrame({'Feature': selected_features}).to_csv('selected_features.csv', index=False)
    importance_df.to_csv('feature_importance.csv', index=False)
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(30)
    
    # Create color map by feature type
    colors = []
    for f_type in top_features['Type']:
        if f_type == 'Numeric':
            colors.append('blue')
        elif f_type == 'Date':
            colors.append('green')
        else:
            colors.append('red')
    
    sns.barplot(x='Importance', y='Feature', hue='Type', data=top_features, palette={'Numeric': 'blue', 'Date': 'green', 'Categorical': 'red'}, legend=False)
    plt.title('Top 30 Features by Importance (Blue=Numeric, Green=Date, Red=Categorical)')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300)
    
    # Plot feature importance by type
    plt.figure(figsize=(10, 6))
    type_importance = importance_df.groupby('Type')['Importance'].sum().reset_index()
    sns.barplot(x='Type', y='Importance', data=type_importance)
    plt.title('Total Feature Importance by Type')
    plt.tight_layout()
    plt.savefig('plots/type_importance.png', dpi=300)
    
    return selected_features, importance_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simplified feature selection')
    parser.add_argument('--dataset', type=str, default='small_dataset.csv',
                        help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    # Load and preprocess data
    X, y, feature_names, y_one_hot, class_names = load_and_preprocess_data(args.dataset)
    
    # Run feature selection
    selected_features, importance_df = simple_feature_selection(X, y, feature_names)
    
    # Print top selected features
    print("\nTop 20 selected features:")
    for i, feature in enumerate(selected_features[:20]):
        print(f"{i+1}. {feature}")
    
    print("\nFeature selection complete!")

if __name__ == "__main__":
    main() 