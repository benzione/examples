import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
from datetime import datetime

# Set TensorFlow options
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define checkpoint callbacks
checkpoint_callback = ModelCheckpoint(
    filepath="model_checkpoint.keras",
    save_weights_only=False,
    save_freq="epoch",
    verbose=1,
)

checkpoint_best_callback = ModelCheckpoint(
    filepath="model_best.keras",
    monitor="accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

def preprocess_dataset(df):
    """
    Preprocess the synthetic dataset:
    - Handle numeric features
    - Transform date features into numeric representations
    - One-hot encode categorical features
    - Use the existing category column as the target
    """
    print(f"Original dataset shape: {df.shape}")
    
    # Identify feature types by column names
    numeric_cols = [col for col in df.columns if col.startswith('feature_') and not ('_date' in col or '_cat' in col)]
    date_cols = [col for col in df.columns if '_date' in col]
    categorical_cols = [col for col in df.columns if '_cat' in col]
    
    print(f"Found {len(numeric_cols)} numeric features, {len(date_cols)} date features, and {len(categorical_cols)} categorical features")
    print(f"Number of unique categories in target: {df['category'].nunique()}")
    
    # Process date features - extract useful components
    for col in date_cols:
        # Extract year, month, day, day of week, day of year
        df[f"{col}_year"] = pd.to_datetime(df[col]).dt.year
        df[f"{col}_month"] = pd.to_datetime(df[col]).dt.month
        df[f"{col}_day"] = pd.to_datetime(df[col]).dt.day
        df[f"{col}_dayofweek"] = pd.to_datetime(df[col]).dt.dayofweek
        df[f"{col}_dayofyear"] = pd.to_datetime(df[col]).dt.dayofyear
        # Drop the original date column after extraction
        df = df.drop(columns=[col])
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=categorical_cols, prefix_sep='_')
    
    # Note: We're using the existing 'category' column now instead of creating a synthetic one
    
    print(f"Dataset shape after preprocessing: {df.shape}")
    return df

def build_model(input_dim, num_classes):
    """Build a neural network model for classification with many classes"""
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    # Increase layer sizes to handle the larger number of categories
    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))

    # Output layer for multi-class classification
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer="adam", 
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )

    model.summary()
    return model

# Main execution flow
def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("dataset.csv")
    print(f"Dataset loaded with shape: {df.shape}")

    # Preprocess the dataset
    df = preprocess_dataset(df)

    # Prepare the target variable
    encoder = OneHotEncoder(sparse_output=False)
    categories_one_hot = encoder.fit_transform(df[["category"]])
    num_classes = len(encoder.categories_[0])
    print(f"Number of classes: {num_classes}")

    # Remove TIK and category from features
    X = df.drop(columns=["TIK", "category"])
    input_dim = X.shape[1]

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = [col for col in X.columns if 
                   not '_' in col or  # Original numeric features
                   col.endswith('_year') or  # Date derived features
                   col.endswith('_month') or
                   col.endswith('_day') or
                   col.endswith('_dayofweek') or
                   col.endswith('_dayofyear')]
    
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, categories_one_hot, test_size=0.2, random_state=42
    )

    print(f"Training with {X_train.shape[1]} features and {num_classes} classes")

    # Build and train the model
    model = build_model(input_dim, num_classes)

    # Adjust training parameters for a larger model
    epochs = 30
    batch_size = 128
    
    print(f"Starting training with {epochs} epochs and batch size {batch_size}")
    
    # Train the model
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint_best_callback, checkpoint_callback],
    )

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")

    # Save the model and preprocessing components
    model.save("synthetic_dataset_model.keras")
    joblib.dump(scaler, "feature_scaler.pkl")
    joblib.dump(encoder, "target_encoder.pkl")
    
    print("Model and preprocessing components saved")

if __name__ == "__main__":
    main()
