import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from logger import logger

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from sqlalchemy import create_engine, text
import joblib

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath="model_checkpoint.keras",
    save_weights_only=False,  # Save full model (architecture + weights)
    save_freq="epoch",  # Save every epoch
    verbose=1,
)

checkpoint_best_callback = ModelCheckpoint(
    filepath="model_best.keras",
    monitor="accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)


def remove_irrelevant_cols(df):
    df = df.drop(df.filter(like="T_IDCUN", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="ANAF", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="MIKUD", axis=1).columns, axis=1)

    drop_cols = [
        "DIRA",
        "MS_FAX",
        "DIRA_ESEK",
        "MISPAR_YESHUT",
        "TEL_ESEK",
        "MS_TELFON",
        "DATE_TIK",
    ]

    return df.drop(drop_cols, axis=1)


def remove_null_cols(df):
    df_cleaned = df.dropna(axis=1, how="all")
    return df_cleaned


def remove_single_value_cols(df):
    return df.loc[:, df.nunique() != 1]


def make_oneot(df):
    onehot_hardcoded = ["YSHUV"]

    onehot_from_data = [col for col in df.columns if col.endswith("_onehot")]

    onehot_columns = onehot_from_data + onehot_hardcoded
    df_encoded = pd.get_dummies(df, columns=onehot_columns, prefix_sep="_")

    return df_encoded


def remove_correlated_features(df, threshold=0.9):
    df = df.apply(pd.to_numeric, errors="coerce")  # Converts non-numeric values to NaN
    df = df.dropna(axis=1, how="all")  # Drop columns that are entirely NaN

    correlation_matrix = df.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    correlated_features = [
        column for column in upper_tri.columns if any(upper_tri[column] > threshold)
    ]
    return df.drop(columns=correlated_features)


def append_y(df):
    df_excel = pd.read_excel("Book1.xlsx")
    df = df.merge(df_excel[["TIK", "category"]], on="TIK", how="left")
    return df.dropna(subset=["category"])


def another_model(input_dim, num_classes):
    # Define the model
    inputs = layers.Input(shape=(input_dim,))

    # Embedding layer (if features are categorical)
    embedded = layers.Embedding(input_dim=1000, output_dim=64)(inputs)

    # Transformer Block
    x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(embedded, embedded)
    x = layers.Dropout(0.3)(x)
    x = layers.LayerNormalization()(x)

    # Combine with dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    # Output layer for multi-class classification
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create the model
    model = models.Model(inputs, outputs)

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Summary of the model architecture
    model.summary()

    return model


def build_model(input_dim, num_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(256, activation="relu"))

    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    return model


logger.info("starting")

servername = "bisqldwhd1"
dbname = "MCH"
engine = create_engine(
    f"mssql+pyodbc://@{servername}/{dbname}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)

recreate_df = True


# init
if False:
    with open("./SQL/master_query.sql", "r", encoding="utf-8") as file:
        sql_script = file.read()
        with engine.connect() as conn:
            conn.execute(text(sql_script))
            conn.commit()


if recreate_df:
    df = pd.read_sql(
        "SELECT top 250000 * FROM [MCH].[SH\hm24].[yaakov_temp_table] WITH (NOLOCK)",
        engine,
    )
    logger.info("sql selection finished")

    df = append_y(df)
    logger.info("append_y finished")

    df = remove_irrelevant_cols(df)
    logger.info("remove_irrelevant_cols finished")

    df = remove_null_cols(df)
    logger.info("remove_null_cols finished")

    df = remove_single_value_cols(df)
    logger.info("remove_single_value_cols finished")

    # df = remove_correlated_features(df, threshold=0.95)
    # logger.info("remove_correlated_features finished")

    logger.info(f"Total rows: {len(df)}")

logger.info(f"Total features before one-hot: {len(df.columns)}")
df = make_oneot(df)
logger.info(f"Total features after one-hot: {len(df.columns)}")


load_existing = False

if load_existing and os.path.exists("model_onehot_encoder.pkl"):
    encoder = joblib.load("model_onehot_encoder.pkl")
else:
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[["category"]])
    joblib.dump(encoder, "model_onehot_encoder.pkl")  # Save encoder


categories_one_hot = encoder.transform(df[["category"]])
num_classes = len(encoder.categories_[0])

df = df.drop(columns=["category"])
input_dim = df.drop(columns=["TIK"]).shape[1]


X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["TIK"]), categories_one_hot, test_size=0.2, random_state=42
)

if load_existing and os.path.exists("model_checkpoint.keras"):
    logger.info("Loading saved model...")
    model = keras.models.load_model("model_checkpoint.keras")
else:
    logger.info("Initializing new model...")
    model = another_model(input_dim, num_classes)  # build_model(input_dim, num_classes)


model.summary()


# Train the model
model.fit(
    X_train,
    y_train,
    epochs=1500,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_best_callback, checkpoint_callback],
)

model.summary()
