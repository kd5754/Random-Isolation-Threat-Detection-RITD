import os
import json
import numpy as np

from data_preprocessing import load_dataset, preprocess_data
from abfo_feature_selector import ABFOFeatureSelector
from vdpc_clustering import VDPC
from federated_train import federated_train
from evaluate import evaluate_model

# -------------------------------------------------------------------------
# Experiment Configuration
# -------------------------------------------------------------------------
DATASET_PATH = "dataset.csv"
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "trained_dbn_model.h5")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "selected_features.json")
CLUSTERS_PATH = os.path.join(ARTIFACT_DIR, "cluster_labels.npy")
EVALUATION_OUTPUT = os.path.join(ARTIFACT_DIR, "evaluation_metrics.json")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------
def run_experiment():
    print("\n============================")
    print("   ZERO-DAY ATTACK DETECTION")
    print("   FULL EXPERIMENT PIPELINE")
    print("============================\n")

    # -------------------------------------------------------------
    # STEP 1: Load Dataset
    # -------------------------------------------------------------
    print("[STEP 1] Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset(DATASET_PATH)

    # -------------------------------------------------------------
    # STEP 2: Preprocess Dataset
    # -------------------------------------------------------------
    print("[STEP 2] Preprocessing dataset...")
    X_train_p, X_test_p = preprocess_data(X_train, X_test)

    # -------------------------------------------------------------
    # STEP 3: Feature Selection using ABFO
    # -------------------------------------------------------------
    print("[STEP 3] Running ABFO feature selection...")

    feature_selector = ABFOFeatureSelector(
        num_features=X_train_p.shape[1],
        population_size=20,
        n_iterations=20
    )

    selected_mask = feature_selector.select_features(X_train_p, y_train)
    selected_features = np.where(selected_mask == 1)[0]

    print(f"[INFO] Selected {len(selected_features)} out of {X_train_p.shape[1]} features")

    # Save feature indices
    with open(FEATURES_PATH, "w") as f:
        json.dump({"selected_features": selected_features.tolist()}, f, indent=4)

    # Reduce dataset
    X_train_fs = X_train_p[:, selected_mask == 1]
    X_test_fs = X_test_p[:, selected_mask == 1]

    # -------------------------------------------------------------
    # STEP 4: VDPC Clustering (Source Domain)
    # -------------------------------------------------------------
    print("[STEP 4] Performing VDPC clustering...")

    vdpc = VDPC(
        percent=2.0,
        max_iterations=10
    )

    cluster_labels = vdpc.fit_predict(X_train_fs)

    # Save clusters
    np.save(CLUSTERS_PATH, cluster_labels)

    print(f"[INFO] VDPC Clustering Completed. Unique Clusters: {np.unique(cluster_labels)}")

    # -------------------------------------------------------------
    # STEP 5: Federated Model Training (DBN)
    # -------------------------------------------------------------
    print("[STEP 5] Training DBN model in Federated Learning setting...")

    model = federated_train(
        X_train_fs, y_train,
        num_clients=3,
        global_rounds=5,
        model_save_path=MODEL_PATH
    )

    print(f"[INFO] Model saved to: {MODEL_PATH}")

    # -------------------------------------------------------------
    # STEP 6: Evaluation
    # -------------------------------------------------------------
    print("[STEP 6] Evaluating model...")

    metrics = evaluate_model(
        model_path=MODEL_PATH,
        test_data_path=DATASET_PATH
    )

    # Save evaluation output (already saved inside evaluate.py)
    print("[INFO] Evaluation pipeline completed successfully.\n")

    print("===== EXPERIMENT FINISHED =====")
    print("All artifacts saved in:", ARTIFACT_DIR)

    return metrics


# -------------------------------------------------------------------------
# RUN PIPELINE
# -------------------------------------------------------------------------
if __name__ == "__main__":
    run_experiment()
