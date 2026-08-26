"""
train.py

Trains a GRU sequence classifier on the preprocessed landmark data,
evaluates on the held-out test set, and saves a confusion matrix.

REQUIRES (in addition to earlier packages):
    pip install scikit-learn matplotlib

USAGE:
    python train.py

INPUT (from data/processed/, created by preprocess_dataset.py):
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    labels.json

OUTPUT:
    models/best_model.keras         -> the best model found during training
    reports/confusion_matrix.png    -> visual confusion matrix on the test set
    reports/training_history.png    -> accuracy/loss curves over training
    Prints final test accuracy and a per-class precision/recall report.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORTS_DIR = "reports"


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with open(os.path.join(DATA_DIR, "labels.json")) as f:
        label_maps = json.load(f)
    index_to_label = label_maps["index_to_label"]
    # JSON keys are strings; convert back to int-keyed, then build an
    # ordered list of class names by index for reporting/plotting.
    num_classes = len(index_to_label)
    class_names = [index_to_label[str(i)] for i in range(num_classes)]

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


def build_model(seq_len, num_features, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(seq_len, num_features)),
        layers.Masking(mask_value=0.0),
        layers.GRU(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def plot_confusion(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved confusion matrix to {out_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_data()
    seq_len, num_features = X_train.shape[1], X_train.shape[2]
    num_classes = len(class_names)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Classes ({num_classes}): {class_names}")

    model = build_model(seq_len, num_features, num_classes)
    model.summary()

    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history, os.path.join(REPORTS_DIR, "training_history.png"))

    print("\nEvaluating on held-out test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=class_names))

    plot_confusion(y_test, y_pred, class_names, os.path.join(REPORTS_DIR, "confusion_matrix.png"))

    print(f"\nBest model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()