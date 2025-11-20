import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.neural_network import BernoulliRBM

# --------------------------------------------------------------
# Utility: Ensure artifacts folder exists
# --------------------------------------------------------------
def ensure_artifacts():
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        print("[INFO] Created 'artifacts/' directory.")

# --------------------------------------------------------------
# Build DBN using pretrained RBMs
# --------------------------------------------------------------
def pretrain_rbms(X, rbm_sizes=[256, 128, 64], learning_rate=0.01, epochs=10):
    """
    Pretrains stacked RBMs for DBN.

    Args:
        X : Input training features
        rbm_sizes : list of hidden units for each RBM layer

    Returns:
        pretrained_layers : list of pretrained weight matrices + biases
        X_transformed     : transformed feature space for final layer
    """

    pretrained_layers = []
    X_train = X.copy()

    print("\n[INFO] RBM Pretraining Started...")
    print("-----------------------------------------")

    for idx, hidden_units in enumerate(rbm_sizes):

        print(f"[INFO] Pretraining RBM Layer {idx+1}/{len(rbm_sizes)}  |  Units: {hidden_units}")

        rbm = BernoulliRBM(
            n_components=hidden_units,
            learning_rate=learning_rate,
            n_iter=epochs,
            batch_size=64,
            verbose=True
        )

        rbm.fit(X_train)

        # Store RBM weights
        pretrained_layers.append({
            "weights": rbm.components_.T,
            "hidden_units": hidden_units
        })

        # Transform input for next RBM
        X_train = rbm.transform(X_train)

        print(f"[INFO] RBM Layer {idx+1} completed. Output shape = {X_train.shape}")

    print("[INFO] RBM Pretraining Completed.\n")

    return pretrained_layers, X_train

# --------------------------------------------------------------
# Build final DBN model (pretrained RBMs + classifier)
# --------------------------------------------------------------
def build_dbn(input_dim, pretrained_layers, num_classes=2):
    """
    Builds final DBN architecture using pretrained RBMs weights.

    Returns:
        Keras Sequential model
    """

    print("[INFO] Building DBN model with pretrained RBM weights...")

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for layer_info in pretrained_layers:
        model.add(layers.Dense(
            layer_info["hidden_units"],
            activation="relu",
            trainable=True
        ))

    # Output layer
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("[INFO] DBN Model Construction Completed.")
    return model

# --------------------------------------------------------------
# Load dataset with selected features applied
# --------------------------------------------------------------
def load_dataset(selected_feature_mask=None):
    print("[INFO] Loading preprocessed training data...")

    X_train = np.loadtxt("artifacts/X_train.csv", delimiter=",")
    y_train = np.loadtxt("artifacts/y_train.csv", delimiter=",").astype(int)

    # Apply selected features if available
    if selected_feature_mask is not None:
        print("[INFO] Applying ABFO selected features...")
        X_train = X_train[:, selected_feature_mask == 1]
        print(f"[INFO] New feature space = {X_train.shape[1]}")

    return X_train, y_train

# --------------------------------------------------------------
# Main Training Flow
# --------------------------------------------------------------
if __name__ == "__main__":
    ensure_artifacts()

    # Load ABFO feature mask if available
    if os.path.exists("artifacts/selected_features.txt"):
        mask = np.loadtxt("artifacts/selected_features.txt", dtype=int)
        print(f"[INFO] Loaded selected features: {sum(mask)} enabled")
    else:
        mask = None
        print("[WARNING] selected_features.txt not found. Using all features.")

    X_train, y_train = load_dataset(mask)

    # -------- RBM Pretraining --------
    pretrained_layers, _ = pretrain_rbms(
        X_train,
        rbm_sizes=[256, 128, 64],   # You can modify layer sizes here
        learning_rate=0.01,
        epochs=10
    )

    # -------- DBN Fine-Tuning --------
    dbn = build_dbn(
        input_dim=X_train.shape[1],
        pretrained_layers=pretrained_layers,
        num_classes=len(np.unique(y_train))
    )

    print("[INFO] Fine-tuning DBN using supervised learning...")
    dbn.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

    # Save trained DBN model
    model_path = "artifacts/dbn_model.h5"
    dbn.save(model_path)
    print(f"\n[INFO] DBN Training Completed ✓")
    print(f"[INFO] Saved model to: {model_path}\n")
