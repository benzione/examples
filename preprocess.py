import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


def preprocess_dataset(df):
    """
    Preprocess the dataset by handling different feature types:
    - Numeric features
    - Date features
    - Categorical features
    
    Args:
        df: pandas DataFrame with raw data
    
    Returns:
        df_processed: pandas DataFrame with processed features
    """
    df_processed = df.copy()
    
    # 1. Process date features
    date_cols = [col for col in df.columns if '_date' in col]
    
    for col in date_cols:
        # Convert to datetime if not already
        df_processed[col] = pd.to_datetime(df_processed[col])
        
        # Extract useful date components
        df_processed[f"{col}_year"] = df_processed[col].dt.year
        df_processed[f"{col}_month"] = df_processed[col].dt.month
        df_processed[f"{col}_day"] = df_processed[col].dt.day
        df_processed[f"{col}_dayofweek"] = df_processed[col].dt.dayofweek
        df_processed[f"{col}_dayofyear"] = df_processed[col].dt.dayofyear
        
    # Drop original date columns after extracting features
    df_processed.drop(columns=date_cols, inplace=True)
    
    # 2. Process categorical features
    cat_cols = [col for col in df.columns if '_cat' in col]
    
    for col in cat_cols:
        # One-hot encode each categorical feature
        dummies = pd.get_dummies(df_processed[col], prefix=col)
        df_processed = pd.concat([df_processed, dummies], axis=1)
    
    # Drop original categorical columns after one-hot encoding
    df_processed.drop(columns=cat_cols, inplace=True)
    
    return df_processed


def train_model(X, y_one_hot, scaler, encoder, save_model=True):
    """
    Train a model on the dataset
    
    Args:
        dataset_path: Path to the dataset CSV file
        save_model: Whether to save the model and preprocessing components
    
    Returns:
        model: Trained model
        X: Features
        y: Target
        feature_names: List of feature names
        class_names: List of class names
    """
    print(f"Dataset loaded with shape: {X.shape}")
    
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    print(f"Training with {X_train.shape[1]} features...")
    
    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(y_one_hot.shape[1], activation="softmax")
    ])
    
    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5,
            monitor="val_loss",
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="synthetic_dataset_model.keras",
            save_best_only=True,
            monitor="val_accuracy"
        )
    ]
    
    # Train the model
    print("Training model...")
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save the model and preprocessing components
    if save_model:
        print("Saving model and preprocessing components...")
        model.save("synthetic_dataset_model.keras")
        joblib.dump(scaler, "feature_scaler.pkl")
        joblib.dump(encoder, "target_encoder.pkl")
    
    return model


if __name__ == "__main__":
    train_model("dataset.csv") 