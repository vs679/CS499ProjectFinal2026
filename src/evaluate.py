import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


NUM_CLASSES = 5

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}


def plot_training_history(history, title):
    """Plot accuracy and loss curves from model training."""
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(7, 4))
    plt.plot(history_df["accuracy"], label="Train Accuracy")
    plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{title} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(history_df["loss"], label="Train Loss")
    plt.plot(history_df["val_loss"], label="Validation Loss")
    plt.title(f"{title} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_predictions(model, generator):
    """Get true labels, predicted labels, and prediction probabilities."""
    probabilities = model.predict(generator, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    true_labels = generator.dataframe["diagnosis"].values[:len(predictions)]

    return true_labels, predictions, probabilities


def evaluate_model(model, generator, model_name):
    """Evaluate a trained model using classification report and confusion matrix."""
    true_labels, predictions, probabilities = get_predictions(model, generator)

    print(f"\nClassification Report for {model_name}")
    print(classification_report(
        true_labels,
        predictions,
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        zero_division=0
    ))

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(NUM_CLASSES))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(ax=ax, xticks_rotation=35)
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.show()

    accuracy = np.mean(true_labels == predictions)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "true_labels": true_labels,
        "predictions": predictions,
        "probabilities": probabilities
    }
