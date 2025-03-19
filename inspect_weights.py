import numpy as np
from tensorflow import keras


def get_weights(model):
    weights, biases = model.layers[0].get_weights()
    abs_weights = np.abs(weights)
    feature_importance = np.mean(abs_weights, axis=1)

    sorted_indices = np.argsort(feature_importance)[::-1]

    for idx in sorted_indices[:20]:
        print(f"Feature {idx}: {feature_importance[idx]}")

    print(f"Average of all: {np.mean(feature_importance)}")
    print(f"Total feature: {len(feature_importance)}")


model = keras.models.load_model("best_model.keras")
get_weights(model)
