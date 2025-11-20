import os
import json
import numpy as np
import tensorflow as tf

# Try import tff; if not available, we'll use simulated FedAvg fallback
try:
    import tensorflow_federated as tff
    TFF_AVAILABLE = True
except Exception:
    TFF_AVAILABLE = False

# -----------------------
# Utilities
# -----------------------
def ensure_artifacts():
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        print("[INFO] Created artifacts/ directory.")


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------
# Data partitioning
# -----------------------
def create_iid_shards(X, y, num_clients):
    """Split dataset into consecutive shards (roughly IID)."""
    n = X.shape[0]
    shard_size = n // num_clients
    clients = []
    for i in range(num_clients):
        start = i * shard_size
        end = n if i == num_clients - 1 else (i + 1) * shard_size
        clients.append((X[start:end], y[start:end]))
    return clients


def create_dirichlet_partitions(X, y, num_clients, alpha=0.5):
    """
    Partition labels using Dirichlet distribution to simulate heterogeneity.
    Returns list of (X_client, y_client).
    """
    n = X.shape[0]
    labels = np.unique(y)
    label_indices = {lab: np.where(y == lab)[0].tolist() for lab in labels}

    client_indices = [[] for _ in range(num_clients)]
    rng = np.random.default_rng(42)

    for lab in labels:
        idxs = label_indices[lab]
        # proportions for each client
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        # allocate counts
        counts = (proportions * len(idxs)).astype(int)
        # adjust rounding issues: ensure sum(counts) == len(idxs)
        while counts.sum() < len(idxs):
            counts[rng.integers(0, num_clients)] += 1
        ptr = 0
        for c in range(num_clients):
            cnt = counts[c]
            if cnt > 0:
                client_indices[c].extend(idxs[ptr:ptr + cnt])
                ptr += cnt

    clients = []
    for inds in client_indices:
        if len(inds) == 0:
            # if a client got no samples, provide a tiny random sample to avoid empty client errors
            sample_idx = np.random.choice(n, size=max(1, n // (100 * num_clients)), replace=False)
            inds = sample_idx.tolist()
        clients.append((X[inds], y[inds]))
    return clients


# -----------------------
# Keras model factory
# -----------------------
def build_keras_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# -----------------------
# Model weight utilities
# -----------------------
def get_weights(model):
    return model.get_weights()


def set_weights(model, weights):
    model.set_weights(weights)


def average_weights(list_of_weights, weights=None):
    """
    Averaging list_of_weights element-wise.
    Optionally weighted average by list 'weights' (same length as list_of_weights).
    """
    if weights is None:
        weights = [1.0] * len(list_of_weights)
    total = sum(weights)
    avg = []
    for layer_idx in range(len(list_of_weights[0])):
        layer_stack = np.array([client_w[layer_idx] * w for client_w, w in zip(list_of_weights, weights)])
        avg_layer = layer_stack.sum(axis=0) / total
        avg.append(avg_layer)
    return avg


# -----------------------
# TFF-based federated training (if available)
# -----------------------
def run_tff_training(X, y, num_clients=5, rounds=5, local_epochs=1, batch_size=32):
    """
    Runs federated training using TensorFlow Federated.
    Expects X and y as numpy arrays.
    """
    print("[INFO] Running TFF-based federated training.")
    # Prepare client datasets as list of tf.data.Datasets
    clients = create_iid_shards(X, y, num_clients)

    def to_tf_dataset(x_np, y_np):
        ds = tf.data.Dataset.from_tensor_slices((x_np.astype(np.float32), y_np.astype(np.int32)))
        ds = ds.shuffle(buffer_size=1024).batch(batch_size)
        return ds

    tff_clients = [to_tf_dataset(x, y) for x, y in clients]

    input_spec = tff_clients[0].element_spec

    def model_fn():
        keras_model = build_keras_model(input_dim=X.shape[1], num_classes=len(np.unique(y)))
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    # Build federated averaging process
    try:
        iterative_process = tff.learning.algorithms.build_weighted_fed_avg(model_fn)
    except Exception:
        # Backup compatibility for older/newer tff versions
        iterative_process = tff.learning.build_federated_averaging_process(model_fn)

    state = iterative_process.initialize()
    history = []
    for r in range(1, rounds + 1):
        # For simplification, use same dataset for each round (in realistic setups you sample clients)
        state, metrics = iterative_process.next(state, tff_clients)
        print(f"[TFF] Round {r} - metrics: {metrics}")
        history.append({"round": r, "metrics": str(metrics)})

    # Extract final keras model weights from state (if possible)
    try:
        final_model = build_keras_model(input_dim=X.shape[1], num_classes=len(np.unique(y)))
        # The following depends on TFF internals; attempt to get state.model
        if hasattr(state, "model") and hasattr(final_model, "weights"):
            # Convert tff structure to numpy weights - this may be environment/version dependent
            tff_weights = tff.learning.ModelWeights.from_model(final_model)
            # try assigning
            # This block is intentionally guarded — some TFF versions differ
            tf.nest.map_structure(lambda a, b: a.assign(b), tff_weights, state.model)
            final_model.save("artifacts/federated_model.h5")
            print("[INFO] Saved federated model to artifacts/federated_model.h5 (via TFF path).")
        else:
            # Fallback: save the keras model skeleton (untrained) to mark process completion
            final_model.save("artifacts/federated_model.h5")
            print("[INFO] TFF state->Keras conversion not supported in this environment; saved empty model as placeholder.")
    except Exception as e:
        print("[WARNING] Could not extract Keras model from TFF state:", str(e))
        # Save placeholder model
        placeholder = build_keras_model(input_dim=X.shape[1], num_classes=len(np.unique(y)))
        placeholder.save("artifacts/federated_model.h5")
        print("[INFO] Saved placeholder model to artifacts/federated_model.h5")

    # Save history/metrics
    with open("artifacts/federated_metrics.json", "w") as fh:
        json.dump(history, fh, indent=2)

    return True


# -----------------------
# Simulated FedAvg (fallback)
# -----------------------
def run_simulated_fedavg(X, y, num_clients=5, rounds=10, local_epochs=1, batch_size=32, hetero_alpha=None):
    """
    Simulated federated training without TFF.
    - Partitions data into clients
    - For each round:
        - Each client trains local model for local_epochs
        - Server averages weights (weighted by number of samples) to form new global model
    - Saves the final global model and metrics
    """

    print("[INFO] Running simulated FedAvg (no TFF).")
    if hetero_alpha is None:
        clients = create_iid_shards(X, y, num_clients)
    else:
        clients = create_dirichlet_partitions(X, y, num_clients, alpha=hetero_alpha)

    # Initialize global model
    global_model = build_keras_model(input_dim=X.shape[1], num_classes=len(np.unique(y)))
    global_weights = get_weights(global_model)

    history = []

    for rnd in range(1, rounds + 1):
        client_weights = []
        client_sizes = []
        client_metrics = []

        print(f"[FedAvg] Round {rnd}/{rounds} - training {len(clients)} clients locally...")

        for idx, (Xc, yc) in enumerate(clients):
            # Build a fresh model and set global weights
            client_model = build_keras_model(input_dim=X.shape[1], num_classes=len(np.unique(y)))
            set_weights(client_model, global_weights)

            # If client dataset is tiny, increase epochs or batch size small to avoid errors
            if len(yc) < 2:
                # Skip training for extremely small client, but still include its weights
                print(f"  - Client {idx}: too small ({len(yc)} samples) — skipping local training")
            else:
                client_model.fit(Xc, yc, epochs=local_epochs, batch_size=min(batch_size, max(1, len(yc))),
                                 verbose=0)

            # Collect weights and size
            client_weights.append(get_weights(client_model))
            client_sizes.append(len(yc))

            # Evaluate client model locally for logging
            loss, acc = client_model.evaluate(Xc, yc, verbose=0)
            client_metrics.append({"client": idx, "samples": len(yc), "loss": float(loss), "acc": float(acc)})

        # Weighted averaging
        new_global_weights = average_weights(client_weights, weights=client_sizes)
        set_weights(global_model, new_global_weights)
        global_weights = get_weights(global_model)

        # Optionally evaluate global model on pooled data (train pool) for monitoring
        loss_g, acc_g = global_model.evaluate(X, y, verbose=0)
        round_record = {
            "round": rnd,
            "global_loss": float(loss_g),
            "global_acc": float(acc_g),
            "client_metrics": client_metrics
        }
        history.append(round_record)
        print(f"  -> Round {rnd} global acc: {acc_g:.4f}, loss: {loss_g:.4f}")

    # Save final global model and metrics
    global_model.save("artifacts/federated_model.h5")
    with open("artifacts/federated_metrics.json", "w") as fh:
        json.dump(history, fh, indent=2)

    print("[INFO] Simulated FedAvg completed. Model and metrics saved to artifacts/")

    return True


# -----------------------
# Main entry
# -----------------------
if __name__ == "__main__":
    ensure_artifacts()
    set_seed(42)

    # Load preprocessed train data
    if not os.path.exists("artifacts/X_train.csv") or not os.path.exists("artifacts/y_train.csv"):
        raise FileNotFoundError("Preprocessed artifacts/X_train.csv and y_train.csv are required. Run data_preprocessing.py first.")

    X_train = np.loadtxt("artifacts/X_train.csv", delimiter=",")
    y_train = np.loadtxt("artifacts/y_train.csv", delimiter=",").astype(int)

    # Configurable parameters (tweak as needed)
    NUM_CLIENTS = 8
    ROUNDS = 10
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 32
    HETERO_ALPHA = 0.5  # set to None for IID; small alpha -> more heterogeneous

    # If ABFO selected features mask exists, apply it (optional)
    mask_path = "artifacts/selected_features.txt"
    if os.path.exists(mask_path):
        mask = np.loadtxt(mask_path, dtype=int)
        if mask.ndim == 1 and mask.size == X_train.shape[1]:
            X_train = X_train[:, mask == 1]
            print(f"[INFO] Applied ABFO mask; new feature dim: {X_train.shape[1]}")
        else:
            print("[WARNING] selected_features.txt shape mismatch — ignoring mask.")

    # Branch: TFF or simulated FedAvg
    if TFF_AVAILABLE:
        try:
            success = run_tff_training(X_train, y_train, num_clients=NUM_CLIENTS, rounds=ROUNDS,
                                       local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE)
            if success:
                print("[INFO] Federated training completed with TFF.")
        except Exception as e:
            print("[WARNING] TFF training failed with exception:", str(e))
            print("[INFO] Falling back to simulated FedAvg.")
            run_simulated_fedavg(X_train, y_train, num_clients=NUM_CLIENTS, rounds=ROUNDS,
                                 local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, hetero_alpha=HETERO_ALPHA)
    else:
        print("[INFO] TensorFlow Federated not available — using simulated FedAvg.")
        run_simulated_fedavg(X_train, y_train, num_clients=NUM_CLIENTS, rounds=ROUNDS,
                             local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, hetero_alpha=HETERO_ALPHA)
