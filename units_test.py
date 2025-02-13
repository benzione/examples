"""
Branch Classification Model
-------------------------
This script implements a machine learning pipeline for classifying business branches using XGBoost.
The model uses a hierarchical approach, first predicting level 1 classifications, then using those
predictions to help predict level 4 (most detailed) classifications.

Key Features:
- Hierarchical classification (level 1 -> level 4)
- Data balancing using RandomUnderSampler
- Feature engineering including ratio calculations
- K-nearest centroids computation
- Cross-validation using folds
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Original imports remain the same...

def masked_cosine_similarity_df(records_df, centroids_df):
    """
    Compute manhattan distances between records and centroids, handling missing values.
    
    Args:
        records_df (pd.DataFrame): DataFrame containing record features
        centroids_df (pd.DataFrame): DataFrame containing centroid features
        
    Returns:
        np.ndarray: Distance matrix between records and centroids
    """
    assert len(records_df.columns) == len(centroids_df.columns), "Columns must match"
    
    records_array = records_df.fillna(0).to_numpy()
    centroids_array = centroids_df.fillna(0).to_numpy()
    records_mask = ~records_df.isna().to_numpy()
    centroids_mask = ~centroids_df.isna().to_numpy()
    
    records_sparse = records_array * records_mask
    centroids_sparse = centroids_array * centroids_mask
    
    return manhattan_distances(records_sparse, centroids_sparse)

def sampling_lmblearn(X, y, up_limit, random_state):
    """
    Balance class distribution by undersampling majority classes.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target labels
        up_limit (int): Maximum samples per class
        random_state (int): Random seed
        
    Returns:
        tuple: (X_resampled, y_resampled) Balanced datasets
    """
    categories_count = y.value_counts()
    target_samples = {k: up_limit for k, _ in categories_count[categories_count > up_limit].to_dict().items() if k in y}
    rus = RandomUnderSampler(sampling_strategy=target_samples, random_state=random_state)
    return rus.fit_resample(X, y)

def metrics_kpis(real_values, pred_values):
    """
    Calculate various classification metrics.
    
    Args:
        real_values (array-like): True labels
        pred_values (array-like): Predicted labels
        
    Returns:
        list: [timestamp, accuracy, precision, recall, f1, mcc, avg_precision, avg_accuracy]
    """
    time = datetime.now()
    metrics = [
        round(accuracy_score(real_values, pred_values), 4),
        round(precision_score(real_values, pred_values, average='macro'), 4),
        round(recall_score(real_values, pred_values, average='macro'), 4),
        round(f1_score(real_values, pred_values, average='macro'), 4),
        round(matthews_corrcoef(real_values, pred_values), 4),
        round(average_precision_score(real_values, pred_values), 4),
        round(balanced_accuracy_score(real_values, pred_values), 4)
    ]
    return [time] + metrics

# Unit Tests
class TestBranchClassification(unittest.TestCase):
    def setUp(self):
        self.records_df = pd.DataFrame({
            'feat1': [1.0, 2.0, np.nan],
            'feat2': [4.0, 5.0, 6.0]
        })
        self.centroids_df = pd.DataFrame({
            'feat1': [1.5, 2.5],
            'feat2': [4.5, 5.5]
        })
        
    def test_masked_cosine_similarity_df(self):
        distances = masked_cosine_similarity_df(self.records_df, self.centroids_df)
        self.assertEqual(distances.shape, (3, 2))
        self.assertTrue(isinstance(distances, np.ndarray))
        
    def test_sampling_lmblearn(self):
        X = pd.DataFrame({'feature': range(10)})
        y = pd.Series([0] * 8 + [1] * 2)
        X_res, y_res = sampling_lmblearn(X, y, up_limit=3, random_state=42)
        self.assertLessEqual(len(X_res), 6)  # 3 samples per class max
        self.assertEqual(len(X_res), len(y_res))
        
    def test_metrics_kpis(self):
        real = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 1, 1])
        metrics = metrics_kpis(real, pred)
        self.assertEqual(len(metrics), 8)  # time + 7 metrics
        self.assertTrue(all(isinstance(m, (float, datetime)) for m in metrics))
        
    @patch('sqlalchemy.create_engine')
    def test_create_connection(self, mock_create_engine):
        engine = create_connection()
        mock_create_engine.assert_called_once()
        self.assertIsNotNone(engine)

if __name__ == '__main__':
    unittest.main()
