#!/usr/bin/env python
"""
Boruta-SHAP feature selection implementation
This combines the Boruta algorithm with SHAP values for feature importance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from preprocess import preprocess_dataset


class BorutaSHAP:
    """
    Implementation of Boruta algorithm using SHAP values for feature importance
    """
    
    def __init__(self, model=None, n_estimators=100, max_iter=20, perc=100, alpha=0.05, random_state=42):
        """
        Initialize the Boruta-SHAP algorithm
        
        Parameters:
        -----------
        model : object
            Pre-trained model to use for SHAP value calculation
        n_estimators : int
            Number of trees in the random forest (used if model is not provided)
        max_iter : int
            Maximum number of iterations to perform
        perc : int
            Percentile to use for shadow feature importance
        alpha : float
            Significance level
        random_state : int
            Random state for reproducibility
        """
        self.model = model
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.perc = perc
        self.alpha = alpha
        self.random_state = random_state
        
        # If no model is provided, create a random forest
        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=7,
                n_jobs=-1, 
                random_state=self.random_state
            )
        
        # Initialize feature status arrays
        self.features_accepted = []
        self.features_rejected = []
        self.features_tentative = []
        self.importance_history = []
        
    def fit(self, X, y):
        """
        Run the Boruta-SHAP algorithm
        
        Parameters:
        -----------
        X : pandas DataFrame
            Feature matrix
        y : array-like
            Target variable
        
        Returns:
        --------
        self : object
            Returns self
        """
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_values = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X
            X = pd.DataFrame(X, columns=feature_names)
        
        # Initialize all features as tentative
        self.features_tentative = feature_names.copy()
        self.features_accepted = []
        self.features_rejected = []
        
        # Create shadow features
        for iteration in range(self.max_iter):
            print(f"Iteration {iteration+1}/{self.max_iter}")
            
            # Break if all features are accepted or rejected
            if len(self.features_tentative) == 0:
                print("All features have been classified. Stopping early.")
                break
            
            # Create shadow features by permuting the values of each feature
            X_shadow = X.copy()
            
            # Debug: Print data types of columns
            print("\nColumn data types:")
            for col in X_shadow.columns:
                print(f"  - {col}: {X_shadow[col].dtype}")
            
            # Create shadow features for all columns
            # We'll handle bool conversion in get_feature_importance
            for col in X.columns:
                X_shadow[f"shadow_{col}"] = X_shadow[col].sample(frac=1, random_state=self.random_state+iteration).values
            
            # Debug: Print shadow feature columns created
            shadow_cols = [col for col in X_shadow.columns if col.startswith('shadow_')]
            print(f"\nCreated {len(shadow_cols)} shadow feature columns")
            
            # Get feature importance using SHAP
            try:
                importance_df = self.get_feature_importance(X_shadow, y)
                self.importance_history.append(importance_df)
                
                # Separate real and shadow feature importances
                real_features = importance_df[~importance_df['Feature'].str.startswith('shadow_')]
                shadow_features = importance_df[importance_df['Feature'].str.startswith('shadow_')]
                
                if len(shadow_features) == 0:
                    print("Warning: No shadow features were created. Check if data contains non-numeric values.")
                    break
                
                # Calculate threshold as the percentile of shadow feature importances
                shadow_threshold = np.percentile(shadow_features['Importance'].values, self.perc)
                
                # Update feature status
                for _, row in real_features.iterrows():
                    feature = row['Feature']
                    importance = row['Importance']
                    
                    # Skip if feature is already accepted or rejected
                    if feature in self.features_accepted or feature in self.features_rejected:
                        continue
                    
                    # Accept features with importance > threshold
                    if importance > shadow_threshold:
                        if feature in self.features_tentative:
                            self.features_tentative.remove(feature)
                        self.features_accepted.append(feature)
                    
                    # Reject features with importance < threshold
                    elif importance < shadow_threshold:
                        if feature in self.features_tentative:
                            self.features_tentative.remove(feature)
                        self.features_rejected.append(feature)
                
                print(f"  Accepted: {len(self.features_accepted)}, Rejected: {len(self.features_rejected)}, Tentative: {len(self.features_tentative)}")
            
            except Exception as e:
                print(f"Error in iteration {iteration+1}: {str(e)}")
                # If we have at least one successful iteration, we can continue
                if len(self.importance_history) > 0:
                    print("Using results from previous successful iterations.")
                    break
                else:
                    raise
        
        return self
    
    def get_feature_importance(self, X, y):
        """
        Calculate feature importance using SHAP values
        
        Parameters:
        -----------
        X : pandas DataFrame
            Feature matrix
        y : array-like
            Target variable
        
        Returns:
        --------
        importance_df : pandas DataFrame
            DataFrame with feature names and importance values
        """
        # Convert boolean columns to float64 instead of dropping them
        X_numeric = X.copy()
        boolean_cols = []
        
        for col in X_numeric.columns:
            if X_numeric[col].dtype == bool:
                boolean_cols.append(col)
                X_numeric[col] = X_numeric[col].astype(float)
            elif X_numeric[col].dtype == 'object':
                print(f"Warning: Column {col} contains object type data and will be excluded from SHAP analysis")
                X_numeric = X_numeric.drop(columns=[col])
                
        if boolean_cols:
            print(f"Converted {len(boolean_cols)} boolean columns to float64")
            
        if X_numeric.shape[1] == 0:
            raise ValueError("No numeric features available for SHAP analysis after filtering")
            
        # Debug: Print the dtypes after conversion
        print("\nNumeric columns data types after conversion:")
        for col in X_numeric.columns:
            print(f"  - {col}: {X_numeric[col].dtype}")
        
        # Use only the external model as requested
        model_to_explain = self.model
        
        # Create SHAP explainer
        print(f"Creating SHAP explainer for model type: {type(model_to_explain).__name__}")
        
        try:
            # Convert DataFrame to numpy array to avoid object dtype issues
            X_numpy = X_numeric.values.astype(np.float64)
            
            # For Keras/TensorFlow models, we need to use a different approach
            if 'keras' in str(type(model_to_explain)).lower():
                print("Using custom permutation importance for Keras model")
                
                # Split into real and shadow features
                real_features = [col for col in X_numeric.columns if not col.startswith('shadow_')]
                shadow_features = [col for col in X_numeric.columns if col.startswith('shadow_')]
                
                # Check if y is already one-hot encoded
                if len(y.shape) == 1 or y.shape[1] == 1:
                    # Need to convert y to one-hot
                    print("Converting target to one-hot encoding")
                    from sklearn.preprocessing import OneHotEncoder
                    encoder = OneHotEncoder(sparse_output=False)
                    # Reshape to 2D array if needed
                    y_2d = y.reshape(-1, 1) if len(y.shape) == 1 else y
                    y_onehot = encoder.fit_transform(y_2d)
                else:
                    # Already one-hot
                    y_onehot = y
                
                # Calculate importance directly with permutation importance
                X_real = X_numeric[real_features]
                
                # Get baseline predictions
                baseline_preds = model_to_explain.predict(X_real.values, verbose=0)
                
                importance_values = np.zeros(len(real_features))
                
                print(f"Calculating permutation importance for {len(real_features)} features")
                for i, feature in enumerate(real_features):
                    # Create a copy with the feature permuted
                    X_permuted = X_real.copy()
                    X_permuted[feature] = X_real[feature].sample(frac=1, random_state=self.random_state).values
                    
                    # Get predictions with permuted feature
                    permuted_preds = model_to_explain.predict(X_permuted.values, verbose=0)
                    
                    # Calculate importance as mean absolute difference in predictions
                    importance_values[i] = np.mean(np.abs(baseline_preds - permuted_preds))
                
                # Create shadow feature importance by copying from original features
                shadow_importance = np.zeros(len(shadow_features))
                for i, shadow_feature in enumerate(shadow_features):
                    # Extract the original feature name by removing the "shadow_" prefix
                    orig_feature = shadow_feature[7:]  # Remove 'shadow_' prefix
                    
                    # Find the index of the original feature
                    if orig_feature in real_features:
                        orig_idx = real_features.index(orig_feature)
                        shadow_importance[i] = importance_values[orig_idx]
                    else:
                        # If original feature not found, assign zero importance
                        shadow_importance[i] = 0
                
                # Combine real and shadow importances
                all_importance = np.concatenate([importance_values, shadow_importance])
                all_features = real_features + shadow_features
                
            else:
                # For tree-based models, use TreeExplainer
                explainer = shap.Explainer(model_to_explain, X_numpy)
                shap_values = explainer(X_numpy)
                
                # For multi-class, take the mean absolute SHAP value across all classes
                if len(shap_values.shape) > 2:
                    all_importance = np.abs(shap_values.values).mean(axis=(0, 2))
                else:
                    all_importance = np.abs(shap_values.values).mean(axis=0)
                
                all_features = X_numeric.columns.tolist()
        except Exception as e:
            print(f"Error creating SHAP explainer with external model: {str(e)}")
            raise
        
        # Create DataFrame with feature names and importance values
        importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': all_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot
        
        Returns:
        --------
        fig : matplotlib figure
            Figure object
        """
        # Get the latest importance values
        if not self.importance_history:
            raise ValueError("No feature importance history available. Run fit() first.")
        
        importance_df = self.importance_history[-1]
        real_features = importance_df[~importance_df['Feature'].str.startswith('shadow_')]
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=real_features.head(top_n))
        plt.title(f'Top {top_n} Feature Importance (SHAP)')
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/boruta_shap_importance.png')
        
        return plt.gcf()


def run_boruta_shap_selection(X, y, feature_names, iterations=20, external_model=None):
    """
    Run Boruta-SHAP feature selection
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : array-like
        Target variable
    feature_names : list
        List of feature names
    iterations : int
        Number of iterations for Boruta
    external_model : object
        Pre-trained model to use for SHAP value calculation
    
    Returns:
    --------
    selected_features : list
        List of selected feature names
    importance_df : pandas DataFrame
        DataFrame with feature names and importance values
    """
    # Convert target to numeric if needed
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)
    else:
        y_numeric = y
    
    # Run Boruta-SHAP
    boruta = BorutaSHAP(model=external_model, max_iter=iterations)
    boruta.fit(X, y_numeric)
    
    # Get feature importances
    importance_df = boruta.importance_history[-1]
    importance_df = importance_df[~importance_df['Feature'].str.startswith('shadow_')]
    
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
    
    # Plot feature importance
    boruta.plot_feature_importance(top_n=min(20, len(selected_features)))
    
    # Save selected features and importance
    pd.DataFrame({'Feature': selected_features}).to_csv('boruta_shap_selected_features.csv', index=False)
    importance_df.to_csv('boruta_shap_feature_importance.csv', index=False)
    
    return selected_features, importance_df
