"""
Manual implementation of Boruta feature selection algorithm.
Use this if you have issues installing the BorutaShap or boruta-py packages.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import shap

class ManualBoruta:
    """
    Manual implementation of Boruta algorithm for feature selection
    """
    
    def __init__(self, n_estimators=100, max_iter=100, perc=100, alpha=0.05, random_state=42):
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
            max_depth=5,
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
        
        Parameters:
        -----------
        X : pandas DataFrame
            Feature dataframe
            
        Returns:
        --------
        X_shadow : pandas DataFrame
            Feature dataframe with shadow features added
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
        
        Parameters:
        -----------
        X : pandas DataFrame
            Feature dataframe
        y : pandas Series or numpy array
            Target variable
            
        Returns:
        --------
        importances : pandas DataFrame
            DataFrame with feature importances
        shadow_max : float
            Maximum importance of shadow features
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
        
        Parameters:
        -----------
        importance_df : pandas DataFrame
            DataFrame with feature importances
        shadow_max : float
            Maximum importance of shadow features
        """
        # Compare each feature importance with shadow max
        for idx, row in importance_df.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            
            # Feature is significantly more important than shadow features
            if importance > shadow_max and feature not in self.features_accepted:
                if feature not in self.features_tentative:
                    self.features_tentative.append(feature)
                
            # Feature is significantly less important than shadow features
            elif importance <= shadow_max and feature not in self.features_rejected and feature not in self.features_accepted:
                if feature in self.features_tentative:
                    self.features_tentative.remove(feature)
                self.features_rejected.append(feature)
        
        # After several iterations, move features from tentative to accepted
        to_accept = []
        for feature in self.features_tentative:
            # Check if feature has been consistently better than shadow features
            if self.importance_history.count(feature) > self.max_iter * 0.7:
                to_accept.append(feature)
                
        # Update accepted features
        for feature in to_accept:
            self.features_accepted.append(feature)
            self.features_tentative.remove(feature)
    
    def fit(self, X, y):
        """
        Run Boruta algorithm
        
        Parameters:
        -----------
        X : pandas DataFrame
            Feature dataframe
        y : pandas Series or numpy array
            Target variable
            
        Returns:
        --------
        self : ManualBoruta
            Fitted object
        """
        print("Running manual Boruta feature selection...")
        
        # Run for specified number of iterations
        for i in range(self.max_iter):
            # Get feature importances
            importance_df, shadow_max = self._get_feature_importances(X, y)
            
            # Update histories
            self.importance_history.extend(importance_df[importance_df['Importance'] > shadow_max]['Feature'].tolist())
            self.shadow_max_history.append(shadow_max)
            
            # Update feature status
            self._update_status(importance_df, shadow_max)
            
            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
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
    
    def transform(self, X):
        """
        Transform dataframe to include only selected features
        
        Parameters:
        -----------
        X : pandas DataFrame
            Feature dataframe
            
        Returns:
        --------
        X_selected : pandas DataFrame
            Transformed dataframe with only selected features
        """
        # Include accepted and tentative features
        selected_features = self.features_accepted + self.features_tentative
        
        return X[selected_features]
    
    def fit_transform(self, X, y):
        """
        Fit and transform
        
        Parameters:
        -----------
        X : pandas DataFrame
            Feature dataframe
        y : pandas Series or numpy array
            Target variable
            
        Returns:
        --------
        X_selected : pandas DataFrame
            Transformed dataframe with only selected features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self):
        """
        Get mask of selected features
        
        Returns:
        --------
        support : numpy array
            Boolean mask of selected features
        """
        selected_features = self.features_accepted + self.features_tentative
        
        return np.array([col in selected_features for col in X.columns])
    
    def get_feature_importance(self, X):
        """
        Get feature importances for all features
        
        Parameters:
        -----------
        X : pandas DataFrame
            Feature dataframe
            
        Returns:
        --------
        importance_df : pandas DataFrame
            DataFrame with feature importance and status
        """
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
    
    def plot_importances(self, X, top_n=30):
        """
        Plot feature importances
        
        Parameters:
        -----------
        X : pandas DataFrame
            Feature dataframe
        top_n : int
            Number of top features to plot
            
        Returns:
        --------
        fig : matplotlib figure
            Figure with feature importances
        """
        # Get feature importances
        importance_df = self.get_feature_importance(X)
        
        # Create figure
        plt.figure(figsize=(10, 12))
        
        # Create color palette based on status
        colors = {'Accepted': 'green', 'Tentative': 'orange', 'Rejected': 'red'}
        palette = [colors[status] for status in importance_df.head(top_n)['Status']]
        
        # Create barplot
        ax = sns.barplot(
            x='Importance', 
            y='Feature', 
            data=importance_df.head(top_n),
            palette=palette
        )
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Accepted'),
            Patch(facecolor='orange', label='Tentative'),
            Patch(facecolor='red', label='Rejected')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        # Save figure
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/manual_boruta_importances.png', dpi=300, bbox_inches='tight')
        
        return plt.gcf()

def run_manual_boruta_shap(X, y, model=None, n_estimators=100, max_iter=100):
    """
    Run manual Boruta and SHAP feature selection
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature dataframe
    y : pandas Series or numpy array
        Target variable
    model : trained model, optional
        Model for SHAP values (if None, will use RandomForest)
    n_estimators : int
        Number of trees in random forest
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    final_features : list
        List of selected features
    importance_df : pandas DataFrame
        DataFrame with feature importances
    """
    # Run manual Boruta
    boruta = ManualBoruta(n_estimators=n_estimators, max_iter=max_iter)
    boruta.fit(X, y)
    
    # Get Boruta selected features
    boruta_features = boruta.features_accepted + boruta.features_tentative
    
    # Get feature importances from Boruta
    boruta_importance = boruta.get_feature_importance(X)
    
    # Calculate SHAP values if model is provided
    if model is not None:
        print("Calculating SHAP values...")
        # For large datasets, use a sample
        if X.shape[0] > 500:
            X_sample = X.sample(500, random_state=42)
        else:
            X_sample = X
            
        # Create background dataset for SHAP
        X_summary = shap.kmeans(X_sample, 50)
        
        # Create explainer
        explainer = shap.KernelExplainer(model.predict, X_summary)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Create a DataFrame with SHAP importance values
        shap_importance = np.zeros(len(X.columns))
        for class_idx in range(len(shap_values)):
            shap_importance += np.abs(shap_values[class_idx]).mean(axis=0)
        
        shap_importance_df = pd.DataFrame({
            'Feature': X.columns.tolist(),
            'SHAP_Importance': shap_importance / len(shap_values)
        }).sort_values('SHAP_Importance', ascending=False)
        
        # Get top features from SHAP
        num_boruta_features = len(boruta_features)
        top_shap_features = shap_importance_df['Feature'].values[:num_boruta_features].tolist()
        
        # Combine features
        final_features = list(set(boruta_features + top_shap_features))
        
        # Merge importance dataframes
        combined_df = pd.merge(
            boruta_importance,
            shap_importance_df,
            on='Feature',
            how='outer'
        )
        combined_df['Final_Selected'] = combined_df['Feature'].isin(final_features)
        
        # Plot combined feature importance
        plt.figure(figsize=(12, 8))
        top_combined = combined_df[combined_df['Final_Selected']].sort_values('SHAP_Importance', ascending=False).head(30)
        
        # Create color map
        colors = ['blue' if row['Status'] == 'Accepted' else 'red' for _, row in top_combined.iterrows()]
        
        # Create bar plot
        sns.barplot(x='SHAP_Importance', y='Feature', data=top_combined, palette=colors)
        plt.title('Top 30 Selected Features (Blue: Boruta Selected, Red: SHAP Added)')
        plt.tight_layout()
        
        # Save figure
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/manual_combined_importances.png', dpi=300, bbox_inches='tight')
        
        # Save results
        combined_df.to_csv('manual_feature_importance.csv', index=False)
        pd.DataFrame({'Feature': final_features}).to_csv('manual_selected_features.csv', index=False)
        
        return final_features, combined_df
    
    else:
        # If no model provided, just return Boruta results
        return boruta_features, boruta_importance

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_breast_cancer
    
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Run manual Boruta
    features, importance = run_manual_boruta_shap(X, y)
    
    print(f"Selected {len(features)} features")
    print("Top 10 features:")
    for i, feature in enumerate(features[:10]):
        print(f"{i+1}. {feature}") 