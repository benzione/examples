import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from BorutaShap import BorutaShap
import seaborn as sns

def load_preprocessed_data(dataset_path, scaler_path, encoder_path):
    """
    Load and preprocess the dataset using saved preprocessing components
    """
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Reuse the preprocessing logic from main.py
    from main import preprocess_dataset
    df_processed = preprocess_dataset(df)
    
    # Load the saved scaler and encoder
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    
    # Prepare the target variable
    categories_one_hot = encoder.transform(df_processed[["category"]])
    categories = df_processed["category"].values
    
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
    
    return X, categories, feature_names, categories_one_hot, encoder.categories_[0]

def boruta_feature_selection(X, y, feature_names, verbose=True, n_estimators=100):
    """
    Perform Boruta feature selection using BorutaShap
    
    Parameters:
    -----------
    X : pandas DataFrame
        Features
    y : numpy array
        Target variable (not one-hot encoded)
    feature_names : list
        List of feature names
    verbose : bool
        Whether to print verbose output
    n_estimators : int
        Number of estimators for the RandomForest
        
    Returns:
    --------
    selected_feature_names : list
        List of selected feature names
    importance_df : pandas DataFrame
        DataFrame with feature importance values
    """
    print("Running Boruta feature selection with BorutaShap...")
    
    # Convert y to pandas Series with proper name for compatibility
    y_series = pd.Series(y, name='target')
    
    # Initialize Random Forest for Boruta
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    
    # Initialize BorutaShap
    # Note: BorutaShap directly computes SHAP values internally
    boruta_shap = BorutaShap(model=rf, importance_measure='shap', classification=True)
    
    # Execute Boruta feature selection 
    boruta_shap.fit(X=X, y=y_series, n_trials=100, sample=False, verbose=verbose)
    
    # Get the feature ranking and selection results
    feature_ranks = boruta_shap.check_features()
    
    # Extracted selected features
    selected_features = feature_ranks[feature_ranks['Decision'] == 'Accepted'].index.tolist()
    tentative_features = feature_ranks[feature_ranks['Decision'] == 'Tentative'].index.tolist()
    
    print(f"Selected {len(selected_features)} features")
    print(f"Tentative features: {len(tentative_features)}")
    
    # Add selection status to the importance dataframe
    feature_ranks['Selected'] = feature_ranks['Decision'] == 'Accepted'
    feature_ranks['Tentative'] = feature_ranks['Decision'] == 'Tentative'
    
    # Rename the column for consistency
    importance_df = feature_ranks.rename(columns={
        'Rank': 'Rank',
        'Mean': 'Mean_Importance',
        'Decision': 'Decision'
    }).reset_index().rename(columns={'index': 'Feature'})
    
    # Sort by rank for better display
    importance_df = importance_df.sort_values('Rank')
    
    if verbose:
        print("Top 20 selected features by Boruta:")
        print(importance_df[importance_df['Selected']].head(20))
    
    return selected_features, importance_df

def calculate_shap_values(model, X, feature_names, class_names, plot=True, sample_size=500):
    """
    Calculate SHAP values for the trained model and create summary plots
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        The trained model
    X : pandas DataFrame
        Features data
    feature_names : list
        List of feature names
    class_names : list
        List of class names
    plot : bool
        Whether to generate and save plots
    sample_size : int
        Number of samples to use for SHAP calculations

    Returns:
    --------
    shap_importance_df : pandas DataFrame
        DataFrame with feature importance based on SHAP values
    """
    print("Calculating SHAP values...")
    
    # If the dataset is large, use a sample
    if X.shape[0] > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X
        
    # Create a background dataset for SHAP
    X_summary = shap.kmeans(X_sample, 50)
    
    # Create explainer
    explainer = shap.KernelExplainer(model.predict, X_summary)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Create a DataFrame with SHAP importance values
    shap_importance = np.zeros(len(feature_names))
    for class_idx in range(len(shap_values)):
        shap_importance += np.abs(shap_values[class_idx]).mean(axis=0)
    
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': shap_importance / len(shap_values)
    }).sort_values('SHAP_Importance', ascending=False)
    
    if plot:
        # Create output directory for plots
        os.makedirs('shap_plots', exist_ok=True)
        
        # Global summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                          class_names=class_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_plots/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar plot of feature importance
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", 
                          feature_names=feature_names, class_names=class_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_plots/shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save top 20 features importance as a bar chart
        plt.figure(figsize=(10, 8))
        sns.barplot(x='SHAP_Importance', y='Feature', 
                    data=shap_importance_df.head(20))
        plt.title('Top 20 Features by SHAP Importance')
        plt.tight_layout()
        plt.savefig('shap_plots/top_features_shap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    return shap_importance_df

def boruta_shap_selection(X, y, y_one_hot, feature_names, class_names, model):
    """
    Feature selection using BorutaShap and additional SHAP values from the model
    
    Parameters:
    -----------
    X : pandas DataFrame
        Features
    y : numpy array
        Target variable (not one-hot encoded)
    y_one_hot : numpy array
        One-hot encoded target variable
    feature_names : list
        List of feature names
    class_names : list
        List of class names
    model : tensorflow.keras.Model
        The trained model
        
    Returns:
    --------
    final_features : list
        List of final selected feature names
    """
    # Run BorutaShap feature selection
    boruta_features, boruta_importance = boruta_feature_selection(X, y, feature_names)
    
    # Also calculate SHAP values on the trained deep learning model for comparison
    shap_importance = calculate_shap_values(model, X, feature_names, class_names)
    
    # Get top features from SHAP (as many as BorutaShap selected)
    num_boruta_features = len(boruta_features)
    top_shap_features = shap_importance['Feature'].values[:num_boruta_features].tolist()
    
    # Combine features from both methods
    final_features = list(set(boruta_features + top_shap_features))
    
    print(f"Selected {len(final_features)} features using Boruta+SHAP approach")
    print(f"BorutaShap selected {len(boruta_features)} features")
    print(f"SHAP from DL model provided {len(top_shap_features)} top features")
    print(f"Final unique features: {len(final_features)}")
    
    # Save the selected features
    pd.DataFrame({'Feature': final_features}).to_csv('selected_features.csv', index=False)
    
    # Create a DataFrame combining both importance scores
    combined_df = pd.merge(
        boruta_importance, 
        shap_importance,
        on='Feature',
        how='outer'
    )
    combined_df['Final_Selected'] = combined_df['Feature'].isin(final_features)
    combined_df.to_csv('feature_importance.csv', index=False)
    
    # Plot combined feature importance
    plt.figure(figsize=(12, 8))
    top_combined = combined_df[combined_df['Final_Selected']].sort_values('SHAP_Importance', ascending=False).head(30)
    
    # Create a color map
    colors = ['blue' if row['Selected'] else 'red' for _, row in top_combined.iterrows()]
    
    # Create bar plot
    sns.barplot(x='SHAP_Importance', y='Feature', data=top_combined, palette=colors)
    plt.title('Top 30 Selected Features (Blue: Boruta Selected, Red: SHAP Added)')
    plt.tight_layout()
    plt.savefig('shap_plots/combined_top_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return final_features, combined_df

def retrain_with_selected_features(X, y_one_hot, final_features, model_architecture, epochs=20, batch_size=128):
    """
    Retrain the model using only the selected features
    
    Parameters:
    -----------
    X : pandas DataFrame
        All features
    y_one_hot : numpy array
        One-hot encoded target variable
    final_features : list
        List of final selected feature names
    model_architecture : str
        Path to saved model architecture
    epochs : int
        Number of epochs for training
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    model : tensorflow.keras.Model
        The retrained model
    """
    from sklearn.model_selection import train_test_split
    
    # Reduce features to selected ones
    X_selected = X[final_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_one_hot, test_size=0.2, random_state=42
    )
    
    # Load the original model to get the architecture
    original_model = keras.models.load_model(model_architecture)
    
    # Create a new model with the same architecture but adjusted input dimension
    input_dim = len(final_features)
    num_classes = y_one_hot.shape[1]
    
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_dim,)))
    
    # Add the same hidden layers as the original model
    for layer in original_model.layers[1:-1]:  # Skip input and output layers
        config = layer.get_config()
        weights = layer.get_weights()
        model.add(keras.layers.Dense.from_config(config))
        # Only set weights if shapes match (they might not due to different input dim)
        if len(model.layers) > 1 and model.layers[-1].weights[0].shape == weights[0].shape:
            model.layers[-1].set_weights(weights)
    
    # Add output layer
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    
    # Compile
    model.compile(
        optimizer="adam", 
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    
    # Define checkpoint for the selected features model
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="model_selected_features_checkpoint.keras",
        save_weights_only=False,
        save_freq="epoch",
        verbose=1,
    )

    checkpoint_best_callback = keras.callbacks.ModelCheckpoint(
        filepath="model_selected_features_best.keras",
        monitor="accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    
    # Train the model
    print(f"Training model with {len(final_features)} selected features...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint_best_callback, checkpoint_callback],
    )
    
    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss with selected features: {score[0]}")
    print(f"Test accuracy with selected features: {score[1]}")
    
    # Save the model
    model.save("synthetic_dataset_model_selected_features.keras")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history_selected_features.png', dpi=300)
    plt.close()
    
    return model, history

def main():
    # Load dataset and preprocessing components
    X, y, feature_names, y_one_hot, class_names = load_preprocessed_data(
        dataset_path="dataset.csv",
        scaler_path="feature_scaler.pkl",
        encoder_path="target_encoder.pkl"
    )
    
    # Load the trained model
    model = keras.models.load_model("synthetic_dataset_model.keras")
    
    # Perform combined feature selection
    final_features, importance_df = boruta_shap_selection(
        X, y, y_one_hot, feature_names, class_names, model
    )
    
    # Print selected features
    print("\nTop 20 selected features:")
    for i, feature in enumerate(final_features[:20]):
        print(f"{i+1}. {feature}")
    
    # Optional: Retrain the model with only the selected features
    retrain = input("\nDo you want to retrain the model with only the selected features? (y/n): ")
    if retrain.lower() == 'y':
        new_model, history = retrain_with_selected_features(
            X, y_one_hot, final_features, "synthetic_dataset_model.keras"
        )
        
        print("\nFeature selection and model retraining complete!")
        print("Check the 'shap_plots' directory for feature importance visualizations")
        print("Check 'selected_features.csv' for the list of selected features")
        print("Check 'feature_importance.csv' for detailed feature importance scores")
    else:
        print("\nFeature selection complete without retraining!")
        print("Check the 'shap_plots' directory for feature importance visualizations")
        print("Check 'selected_features.csv' for the list of selected features")
        print("Check 'feature_importance.csv' for detailed feature importance scores")

if __name__ == "__main__":
    main() 