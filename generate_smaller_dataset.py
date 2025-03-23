import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_smaller_dataset(num_samples=500, k=10, m=3, n=5, num_categories=20):
    """
    Generate a smaller synthetic dataset with:
    - TIK: Company serial number
    - k numeric features
    - m date features
    - n categorical features
    - A category label with num_categories unique values
    """
    # Generate TIK (company serial numbers)
    tik = np.arange(1, num_samples + 1)
    
    # Generate numeric features
    numeric_features = {}
    for i in range(k):
        numeric_features[f'feature_{i}'] = np.random.normal(100, 25, num_samples)
    
    # Generate date features
    date_features = {}
    base_date = datetime(2020, 1, 1)
    for i in range(m):
        random_days = np.random.randint(0, 1000, num_samples)
        dates = [base_date + timedelta(days=int(x)) for x in random_days]
        date_features[f'feature_{i}_date'] = dates
    
    # Generate categorical features
    categorical_features = {}
    categories = ['A', 'B', 'C', 'D', 'E']
    for i in range(n):
        categorical_features[f'feature_{i}_cat'] = np.random.choice(categories, num_samples)
    
    # Generate category labels (with cardinality of num_categories)
    category_labels = [f"CAT_{i}" for i in range(num_categories)]
    category_values = np.random.choice(category_labels, size=num_samples)
    
    # Combine all features into a DataFrame
    df = pd.DataFrame({'TIK': tik})
    
    # Add numeric features
    for col, values in numeric_features.items():
        df[col] = values
        
    # Add date features
    for col, values in date_features.items():
        df[col] = values
        
    # Add categorical features
    for col, values in categorical_features.items():
        df[col] = values
    
    # Add the category label column
    df['category'] = category_values
    
    return df

# Generate smaller dataset
df = generate_smaller_dataset(num_samples=500, k=10, m=3, n=5, num_categories=20)
print(f"Smaller dataset shape: {df.shape}")
print("\nSample of generated data:")
print(df.head())
print(f"\nNumber of unique categories: {df['category'].nunique()}")
df.to_csv("small_dataset.csv", index=False) 