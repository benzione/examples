#!/usr/bin/env python
"""
Simple Boruta feature selection that handles all feature types
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from preprocess import preprocess_dataset

class SimpleBoruta:
    """
    Implementation of Boruta algorithm for feature selection
    """
    
    def __init__(self, n_estimators=100, max_iter=20, perc=100, alpha=0.05, random_state=42):
        """
        Initialize the Boruta algorithm
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the random forest
        max_iter : int
            Maximum number of iterations to perform
        perc : int
            Percentile to use for shadow feature importance
        alpha : float
            Significance level
        random_state : int
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.perc = perc
        self.alpha = alpha
        self.random_state = random_state
        
        # Create random forest
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=7,
            n_jobs=-1, 
            random_state=self.random_state
        )
        
        # Initialize results
        self.features_accepted = []
        self.features_rejected = []
        self.features_tentative = []
        self.importance_history = []
        self.shadow_max_history = []

    def _add_shadow_features(self, X):
        """
        Add shadow features (permuted copies of original features)
        """
        # Create shadow features by permuting the original features
        X_shadow = X.copy()
        
        for col in X.columns:
            # Create a shadow feature by permuting values in the column
            X_shadow[f'shadow_{col}'] = np.random.permutation(X[col].values)
            
        return X_shadow
    
    def _get_feature_importances(self, X, y):
        """
        Calculate feature importances using random forest
        """
        # Add shadow features
        X_shadow = self._add_shadow_features(X)
        
        # Train random forest
        self.model.fit(X_shadow, y)
        
        # Get feature importances
        feature_importances = self.model.feature_importances_
        
        # Get original feature names and shadow feature names
        feature_names = X.columns.tolist()
        shadow_names = [f'shadow_{col}' for col in feature_names]
        all_names = feature_names + shadow_names
        
        # Create dataframe with feature importances
        importance_df = pd.DataFrame({
            'Feature': all_names,
            'Importance': feature_importances
        })
        
        # Split into original and shadow importances
        original_importances = importance_df[importance_df['Feature'].isin(feature_names)]
        shadow_importances = importance_df[importance_df['Feature'].isin(shadow_names)]
        
        # Get maximum shadow importance
        shadow_max = shadow_importances['Importance'].max()
        
        return original_importances, shadow_max
    
    def _update_status(self, importance_df, shadow_max):
        """
        Update feature status (accepted, rejected, tentative)
        """
        # Use a more liberal threshold - 90% of shadow max
        shadow_threshold = shadow_max * 0.9
        
        # Compare each feature importance with shadow max
        for idx, row in importance_df.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            
            # Feature is significantly more important than shadow features
            if importance > shadow_threshold and feature not in self.features_accepted:
                if feature not in self.features_tentative:
                    self.features_tentative.append(feature)
                
            # Feature is significantly less important than shadow features
            elif importance <= shadow_threshold and feature not in self.features_rejected and feature not in self.features_accepted:
                if feature in self.features_tentative:
                    self.features_tentative.remove(feature)
                self.features_rejected.append(feature)
        
        # After several iterations, move features from tentative to accepted
        to_accept = []
        for feature in self.features_tentative:
            # Lower the threshold to accept features - 40% of iterations
            feature_hits = self.importance_history.count(feature)
            if feature_hits > self.max_iter * 0.4:  # Accept if feature is important in 40% of iterations
                to_accept.append(feature)
                
        # Update accepted features
        for feature in to_accept:
            self.features_accepted.append(feature)
            self.features_tentative.remove(feature)
    
    def fit(self, X, y):
        """
        Run Boruta algorithm
        """
        print("Running Boruta feature selection...")
        
        # Store the target variable for later use
        self.y = y
        
        # Run for specified number of iterations
        for i in range(self.max_iter):
            # Get feature importances
            importance_df, shadow_max = self._get_feature_importances(X, y)
            
            # Save the last iteration's importance and shadow_max for visualization
            if i == self.max_iter - 1:
                self.last_importance_df = importance_df
                self.last_shadow_max = shadow_max
            
            # Update histories
            self.importance_history.extend(importance_df[importance_df['Importance'] > shadow_max * 0.9]['Feature'].tolist())
            self.shadow_max_history.append(shadow_max)
            
            # Update feature status
            self._update_status(importance_df, shadow_max)
            
            # Print progress every 5 iterations
            if (i + 1) % 5 == 0:
                print(f"Iteration {i + 1}/{self.max_iter}")
                print(f"  - Accepted features: {len(self.features_accepted)}")
                print(f"  - Tentative features: {len(self.features_tentative)}")
                print(f"  - Rejected features: {len(self.features_rejected)}")
        
        # Final report
        print("\nBoruta feature selection complete.")
        print(f"  - Accepted features: {len(self.features_accepted)}")
        print(f"  - Tentative features: {len(self.features_tentative)}")
        print(f"  - Rejected features: {len(self.features_rejected)}")
        
        return self

    def get_feature_importance(self, X, y=None):
        """
        Get feature importances for all features
        """
        # Use stored y if not provided
        if y is None:
            if hasattr(self, 'y'):
                y = self.y
            else:
                raise ValueError("Target variable 'y' not provided and not stored from previous fit")
                
        # Calculate final feature importances
        self.model.fit(X, y)
        importances = self.model.feature_importances_
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': X.columns.tolist(),
            'Importance': importances,
            'Status': ['Accepted' if col in self.features_accepted 
                      else 'Tentative' if col in self.features_tentative 
                      else 'Rejected' for col in X.columns]
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df

    def plot_shadow_comparison(self, X, save_path="plots/shadow_comparison.png"):
        """
        Plot feature importance compared to shadow features
        """
        if not hasattr(self, 'last_importance_df') or not hasattr(self, 'last_shadow_max'):
            print("Need to run fit first to generate shadow comparison data")
            return
            
        # Add feature type information
        self.last_importance_df['Type'] = self.last_importance_df['Feature'].apply(lambda f: 
            'Numeric' if f.startswith('feature_') and not any(suffix in f for suffix in ['_cat_', '_year', '_month', '_day', '_dayofweek', '_dayofyear']) else
            'Date' if any(suffix in f for suffix in ['_year', '_month', '_day', '_dayofweek', '_dayofyear']) else
            'Categorical'
        )
        
        # Add selection status
        self.last_importance_df['Status'] = self.last_importance_df['Feature'].apply(lambda f:
            'Accepted' if f in self.features_accepted else
            'Tentative' if f in self.features_tentative else
            'Rejected'
        )
        
        # Sort by importance
        sorted_df = self.last_importance_df.sort_values('Importance', ascending=False)
        
        # Get top features
        top_features = sorted_df.head(30)
        
        # Plot feature importance with shadow max line
        plt.figure(figsize=(14, 10))
        
        # Create color palette based on selection status
        status_colors = {'Accepted': 'green', 'Tentative': 'orange', 'Rejected': 'red'}
        hue_order = ['Accepted', 'Tentative', 'Rejected']
        
        # Plot bar chart
        ax = sns.barplot(
            x='Importance', 
            y='Feature', 
            hue='Status',
            hue_order=hue_order,
            palette=status_colors,
            data=top_features
        )
        
        # Add shadow max line
        shadow_threshold = self.last_shadow_max * 0.9  # Same threshold as used in the algorithm
        plt.axvline(x=shadow_threshold, color='blue', linestyle='--', 
                    label=f'Shadow Threshold (90% of max shadow importance)')
        plt.axvline(x=self.last_shadow_max, color='red', linestyle='--', 
                    label='Shadow Max')
        
        plt.title('Feature Importance vs Shadow Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Shadow comparison plot saved to {save_path}")
        
        return plt.gcf()

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
    y = df_processed["category"].values
    
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
    
    return X, y, feature_names

def run_boruta_selection(X, y, feature_names, iterations=20):
    """
    Run Boruta feature selection
    """
    # Convert target to numeric
    le = LabelEncoder()
    y_numeric = le.fit_transform(y)
    
    # Run Boruta
    boruta = SimpleBoruta(n_estimators=100, max_iter=iterations)
    boruta.fit(X, y_numeric)
    
    # Get feature importances
    importance_df = boruta.get_feature_importance(X, y_numeric)
    
    # Get selected features (accepted + tentative)
    selected_features = boruta.features_accepted + boruta.features_tentative
    
    print(f"\nSelected {len(selected_features)} features")
    print(f"  - Accepted features: {len(boruta.features_accepted)}")
    print(f"  - Tentative features: {len(boruta.features_tentative)}")
    
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
    
    # Add feature type information
    importance_df['Type'] = importance_df['Feature'].apply(lambda f: 
        'Numeric' if f.startswith('feature_') and not any(suffix in f for suffix in ['_cat_', '_year', '_month', '_day', '_dayofweek', '_dayofyear']) else
        'Date' if any(suffix in f for suffix in ['_year', '_month', '_day', '_dayofweek', '_dayofyear']) else
        'Categorical'
    )
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Save selected features and importance
    pd.DataFrame({'Feature': selected_features}).to_csv('boruta_selected_features.csv', index=False)
    importance_df.to_csv('boruta_feature_importance.csv', index=False)
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(30)
    
    sns.barplot(x='Importance', y='Feature', hue='Type', 
                data=top_features, 
                palette={'Numeric': 'blue', 'Date': 'green', 'Categorical': 'red'})
    plt.title('Top 30 Features by Importance')
    plt.tight_layout()
    plt.savefig('plots/boruta_feature_importance.png', dpi=300)
    
    # Plot feature importance by status
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', hue='Status', 
                data=top_features.sort_values('Importance', ascending=False).head(30),
                palette={'Accepted': 'green', 'Tentative': 'orange', 'Rejected': 'red'})
    plt.title('Top 30 Features by Status')
    plt.tight_layout()
    plt.savefig('plots/boruta_feature_status.png', dpi=300)
    
    # Create shadow comparison visualization
    boruta.plot_shadow_comparison(X, save_path="plots/shadow_comparison.png")
    
    return selected_features, importance_df

def main():
    parser = argparse.ArgumentParser(description='Run simplified Boruta feature selection')
    parser.add_argument('--dataset', type=str, default='small_dataset.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Number of iterations for Boruta')
    args = parser.parse_args()
    
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data(args.dataset)
    
    # Run feature selection
    selected_features, importance_df = run_boruta_selection(X, y, feature_names, iterations=args.iterations)
    
    # Print top selected features
    print("\nTop 20 selected features:")
    for i, feature in enumerate(selected_features[:20]):
        print(f"{i+1}. {feature}")
    
    print("\nFeature selection complete!")

if __name__ == "__main__":
    main() 