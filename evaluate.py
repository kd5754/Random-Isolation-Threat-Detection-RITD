import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from tensorflow.keras.models import load_model
from data_preprocessing import load_dataset


# -------------------------------------------------------------------------
# Utility: Plot Confusion Matrix
# -------------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Print in each cell
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------------------------
# Evaluation Function
# -------------------------------------------------------------------------
def evaluate_model(model_path, test_data_path):
    """
    Loads trained DBN model and evaluates on test dataset.

    Parameters:
    -----------
    model_path : str
        Path to saved trained DBN model (.h5 file)
    test_data_path : str
        Path to test dataset file

    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """

    # Load dataset
    print("[INFO] Loading test dataset...")
    X_train, X_test, y_train, y_test = load_dataset(test_data_path)

    # Load trained DBN Model
    print("[INFO] Loading trained model...")
    model = load_model(model_path)

    # Predict Probabilities and Class Labels
    print("[INFO] Running predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Compute Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1-Score": round(float(f1), 4),
        "Confusion_Matrix": cm.tolist()
    }

    # ---------------------------------------------------------------------
    # Save Results
    # ---------------------------------------------------------------------
    os.makedirs("artifacts", exist_ok=True)

    with open("artifacts/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plot_confusion_matrix(
        cm,
        class_names=["Normal", "Attack"],
        save_path="artifacts/confusion_matrix.png"
    )

    print("\n[✓] Evaluation complete.")
    print("[✓] Metrics saved to artifacts/evaluation_metrics.json")
    print("[✓] Confusion matrix saved to artifacts/confusion_matrix.png\n")

    return metrics


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_PATH = "artifacts/trained_dbn_model.h5"
    TEST_DATA_PATH = "dataset.csv"

    evaluate_model(MODEL_PATH, TEST_DATA_PATH)
