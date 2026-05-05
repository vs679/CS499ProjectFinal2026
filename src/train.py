import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from preprocessing import FundusDataGenerator, load_and_preprocess_image
from model import build_baseline_cnn, build_transfer_model, fine_tune_model
from evaluate import evaluate_model, plot_training_history


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5

FAST_DEBUG = False

BASELINE_EPOCHS = 15
TRANSFER_EPOCHS_FROZEN = 10
TRANSFER_EPOCHS_FINE_TUNE = 10

if FAST_DEBUG:
    BASELINE_EPOCHS = 2
    TRANSFER_EPOCHS_FROZEN = 1
    TRANSFER_EPOCHS_FINE_TUNE = 1

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}


def set_random_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def locate_dataset():
    """Locate dataset on Kaggle or local machine."""
    kaggle_base = Path("/kaggle/input/aptos2019-blindness-detection")

    if kaggle_base.exists():
        csv_path = kaggle_base / "train.csv"
        image_dir = kaggle_base / "train_images"
    else:
        csv_path = Path("train.csv")
        image_dir = Path("train_images")

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find train.csv at {csv_path}")

    if not image_dir.exists():
        raise FileNotFoundError(f"Could not find image directory at {image_dir}")

    return csv_path, image_dir


def make_callbacks(model_name):
    os.makedirs("models", exist_ok=True)

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            filepath=f"models/{model_name}.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]


def load_dataset(csv_path, image_dir):
    df = pd.read_csv(csv_path)

    df["image_path"] = df["id_code"].apply(
        lambda image_id: str(image_dir / f"{image_id}.png")
    )

    print(df.head())
    print("Total images:", len(df))
    print(df["diagnosis"].value_counts().sort_index())

    return df


def plot_class_distribution(df):
    plt.figure(figsize=(7, 4))
    df["diagnosis"].value_counts().sort_index().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("DR Severity Class")
    plt.ylabel("Number of Images")
    plt.xticks(
        ticks=range(NUM_CLASSES),
        labels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        rotation=35,
        ha="right"
    )
    plt.tight_layout()
    plt.show()


def split_dataset(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["diagnosis"],
        random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["diagnosis"],
        random_state=SEED
    )

    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))

    return train_df, val_df, test_df


def compute_class_weights(train_df):
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sorted(train_df["diagnosis"].unique())),
        y=train_df["diagnosis"]
    )

    class_weights = {
        class_id: weight
        for class_id, weight in enumerate(class_weight_values)
    }

    print("Class weights:")
    print(class_weights)

    return class_weights


def train_baseline(train_df, val_df, test_df, class_weights):
    baseline_train_gen = FundusDataGenerator(
        train_df,
        augment=True,
        model_type="baseline"
    )

    baseline_val_gen = FundusDataGenerator(
        val_df,
        shuffle=False,
        augment=False,
        model_type="baseline"
    )

    baseline_test_gen = FundusDataGenerator(
        test_df,
        shuffle=False,
        augment=False,
        model_type="baseline"
    )

    baseline_model = build_baseline_cnn()
    baseline_model.summary()

    start_time = time.time()

    baseline_history = baseline_model.fit(
        baseline_train_gen,
        validation_data=baseline_val_gen,
        epochs=BASELINE_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks("baseline_cnn"),
        verbose=1
    )

    training_minutes = (time.time() - start_time) / 60
    print(f"Baseline training time: {training_minutes:.2f} minutes")

    plot_training_history(baseline_history, "Baseline CNN")
    baseline_results = evaluate_model(
        baseline_model,
        baseline_test_gen,
        "Baseline CNN"
    )

    return baseline_results, training_minutes


def train_transfer_model(
    model_label,
    model_type,
    train_df,
    val_df,
    test_df,
    class_weights
):
    train_gen = FundusDataGenerator(
        train_df,
        augment=True,
        model_type=model_type
    )

    val_gen = FundusDataGenerator(
        val_df,
        shuffle=False,
        augment=False,
        model_type=model_type
    )

    test_gen = FundusDataGenerator(
        test_df,
        shuffle=False,
        augment=False,
        model_type=model_type
    )

    model, base_model = build_transfer_model(model_label)
    model.summary()

    start_time = time.time()

    history_frozen = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=TRANSFER_EPOCHS_FROZEN,
        class_weight=class_weights,
        callbacks=make_callbacks(f"{model_label}_frozen"),
        verbose=1
    )

    model = fine_tune_model(
        model,
        base_model,
        trainable_layers=30,
        learning_rate=1e-5
    )

    history_finetune = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=TRANSFER_EPOCHS_FINE_TUNE,
        class_weight=class_weights,
        callbacks=make_callbacks(f"{model_label}_finetuned"),
        verbose=1
    )

    training_minutes = (time.time() - start_time) / 60
    print(f"{model_label} total training time: {training_minutes:.2f} minutes")

    plot_training_history(history_frozen, f"{model_label} Frozen Base")
    plot_training_history(history_finetune, f"{model_label} Fine Tuned")

    results = evaluate_model(model, test_gen, model_label)

    return results, training_minutes


def main():
    set_random_seeds()

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    csv_path, image_dir = locate_dataset()

    print("CSV path:", csv_path)
    print("Image directory:", image_dir)

    df = load_dataset(csv_path, image_dir)
    plot_class_distribution(df)

    train_df, val_df, test_df = split_dataset(df)

    class_weights = compute_class_weights(train_df)

    baseline_results, baseline_time = train_baseline(
        train_df,
        val_df,
        test_df,
        class_weights
    )

    resnet_results, resnet_time = train_transfer_model(
        "resnet50",
        "resnet",
        train_df,
        val_df,
        test_df,
        class_weights
    )

    efficientnet_results, efficientnet_time = train_transfer_model(
        "efficientnetb0",
        "efficientnet",
        train_df,
        val_df,
        test_df,
        class_weights
    )

    results_summary = pd.DataFrame([
        {
            "Model": "Baseline CNN",
            "Test Accuracy": baseline_results["accuracy"],
            "Training Time Minutes": baseline_time
        },
        {
            "Model": "ResNet50",
            "Test Accuracy": resnet_results["accuracy"],
            "Training Time Minutes": resnet_time
        },
        {
            "Model": "EfficientNetB0",
            "Test Accuracy": efficientnet_results["accuracy"],
            "Training Time Minutes": efficientnet_time
        }
    ])

    print(results_summary)

    plt.figure(figsize=(8, 4))
    plt.bar(results_summary["Model"], results_summary["Test Accuracy"])
    plt.title("Test Accuracy by Model")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(results_summary["Model"], results_summary["Training Time Minutes"])
    plt.title("Training Time by Model")
    plt.xlabel("Model")
    plt.ylabel("Minutes")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
