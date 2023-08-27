import tensorflow as tf
from RandomForestClassifierTensorFlow import RandomForestClassifierTensorFlow
from SVMClassifierTensorFlow import SVMClassifierTensorFlow
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from math import sqrt


def make_NN_model(
    feature_cols, output_cols, metrics, seed=42, n_layers=5, n_nodes=180
):
    tf.keras.utils.set_random_seed(seed)
    hidden_layers_shape = [n_nodes] * n_layers
    n_inputs = len(feature_cols)
    n_outputs = len(output_cols)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_inputs, activation="relu"))
    for layer in hidden_layers_shape:
        model.add(tf.keras.layers.Dense(layer, activation="relu"))
    model.add(tf.keras.layers.Dense(n_outputs, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=metrics,
    )
    return model


def make_RF_model(
    feature_cols,
    output_cols,
    metrics="",
    seed=42,
    n_estimators=100,
    max_features=None,
):
    if max_features is None:
        max_features = sqrt(len(feature_cols)) / len(feature_cols)
    rf_classifier = RandomForestClassifierTensorFlow(
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            criterion="entropy",
            random_state=seed,
            n_jobs=4,
        )
    )
    return rf_classifier


def make_SVM_model(
    feature_cols,
    output_cols,
    metrics="",
    seed=42,
    C=1.0,
):
    rf_classifier = SVMClassifierTensorFlow(
        LinearSVC(
            C=C,
            dual=False,
            random_state=seed,
        )
    )
    return rf_classifier


def make_NN_model_regression_feature_munger(
    feature_cols, output_cols, metrics, seed=42, n_layers=5, n_nodes=180
):
    tf.keras.utils.set_random_seed(seed)
    hidden_layers_shape = [n_nodes] * n_layers
    n_inputs = len(feature_cols)
    n_outputs = len(output_cols)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_inputs, activation="relu"))
    for layer in hidden_layers_shape:
        model.add(tf.keras.layers.Dense(layer, activation="relu"))
    model.add(tf.keras.layers.Dense(n_outputs, activation="linear"))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=metrics,
    )
    return model
