import os
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def ensure_artifacts():
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        print("[INFO] Created 'artifacts/' directory.")

class VDPC:
    def __init__(
        self,
        k=20,
        sigma=None,
        rho_percentile=85.0,
        delta_percentile=85.0,
        min_cluster_size=10,
        use_scaler=True,
        random_state=42,
    ):
        """
        Parameters
        ----------
        k : int
            Number of neighbors used for local density estimation.
        sigma : float or None
            Gaussian kernel width. If None, sigma = median(kth_distance).
        rho_percentile : float (0-100)
            Percentile to select density peaks by rho*delta or separate thresholds.
        delta_percentile : float (0-100)
            Percentile to threshold delta (distance to higher density).
        min_cluster_size : int
            Minimum number of points for a cluster to be kept; smaller clusters become noise.
        use_scaler : bool
            Whether to scale features with StandardScaler before computing distances.
        random_state : int
            Seed for reproducibility where applicable.
        """
        self.k = int(k)
        self.sigma = sigma
        self.rho_percentile = float(rho_percentile)
        self.delta_percentile = float(delta_percentile)
        self.min_cluster_size = int(min_cluster_size)
        self.use_scaler = bool(use_scaler)
        self.random_state = int(random_state)

        self.labels_ = None
        self.centers_ = None
        self.rho_ = None
        self.delta_ = None
        self.neigh_idx_ = None
        self.metadata_ = {}

    def _estimate_density(self, X):
        """
        Estimate local density rho for each sample using a Gaussian kernel
        with width self.sigma. If sigma is None, estimate sigma from distances.
        Uses kth-nearest neighbor distances as scale reference.
        """
        n_samples = X.shape[0]
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm="auto").fit(X)
        distances, indices = nbrs.kneighbors(X)  # distances: n x (k+1), first col is 0
        # discard self-distance (0)
        kth_dist = distances[:, -1]  # distance to k-th neighbor
        if self.sigma is None:
            # robust scale: median of kth distances
            sigma = np.median(kth_dist)
            if sigma <= 0:
                sigma = np.mean(kth_dist) + 1e-8
        else:
            sigma = self.sigma

        # Gaussian kernel density estimate: sum_i exp(- (d_ij^2) / (2*sigma^2))
        # We can approximate density by using k neighbors only to save cost.
        # Use distances[:, 1:] (exclude self 0)
        distances_no_self = distances[:, 1:]
        rho = np.exp(-(distances_no_self ** 2) / (2.0 * sigma ** 2)).sum(axis=1)
        return rho, distances, indices, sigma

    def _compute_delta(self, X, rho, distances, indices):
        """
        For each point, delta = distance to nearest point with higher density.
        For the highest-density point, delta = max distance in dataset.
        We'll compute pairwise distances to higher-density candidates via neighbor indices
        first; if no higher-density neighbor is found in k-NN, we'll fall back to global search.
        """
        n = X.shape[0]
        # initialize with large value
        delta = np.full(n, -1.0)
        nearest_higher = np.full(n, -1, dtype=int)
        # For efficient lookup, create order of indices sorted by rho descending
        order = np.argsort(-rho)  # indices of points sorted by decreasing density
        # For the highest rho point
        delta[order[0]] = np.max(distances)  # a large distance (max of kNN distances)
        nearest_higher[order[0]] = -1

        # Build a KDTree-like neighbor search for fallback
        nbrs_global = NearestNeighbors(n_neighbors=min(50, n), algorithm="auto").fit(X)

        # For remaining points, search among neighbors then fallback
        for idx in order[1:]:
            # candidates with higher density
            higher_mask = rho > rho[idx]
            if higher_mask.sum() == 0:
                # shouldn't happen because idx is not the highest, but guard
                delta[idx] = np.max(distances)
                nearest_higher[idx] = -1
                continue

            # First try k-NN neighbors for candidate with higher density
            neighs = indices[idx, 1:]  # neighbors (exclude self)
            cand = [j for j in neighs if rho[j] > rho[idx]]
            if len(cand) > 0:
                # compute actual distances to those candidates
                dists = np.linalg.norm(X[cand] - X[idx], axis=1)
                min_pos = np.argmin(dists)
                delta[idx] = dists[min_pos]
                nearest_higher[idx] = cand[min_pos]
            else:
                # Fallback: search globally for nearest higher-density point (use partial nbrs)
                # Use global neighbor structure
                dists_all, idxs_all = nbrs_global.kneighbors(X[idx:idx+1], n_neighbors=min(50, n))
                found = False
                for d, j in zip(dists_all[0], idxs_all[0]):
                    if rho[j] > rho[idx]:
                        delta[idx] = d
                        nearest_higher[idx] = j
                        found = True
                        break
                if not found:
                    # ultimate fallback: brute-force search
                    higher_idxs = np.where(higher_mask)[0]
                    all_dists = np.linalg.norm(X[higher_idxs] - X[idx], axis=1)
                    minpos = np.argmin(all_dists)
                    delta[idx] = all_dists[minpos]
                    nearest_higher[idx] = higher_idxs[minpos]

        return delta, nearest_higher

    def fit(self, X):
        """
        Fit the VDPC model on the feature matrix X.

        After fitting, attributes available:
          - labels_ : np.array of shape (n_samples,) cluster assignments (-1 = noise)
          - centers_: np.array of center indices
          - rho_, delta_ : density and delta arrays
          - metadata_ : dict with params and computed stats
        """
        ensure_artifacts()
        np.random.seed(self.random_state)

        X_proc = X.copy()
        if self.use_scaler:
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(X_proc)
            self.metadata_["scaled"] = True
        else:
            self.metadata_["scaled"] = False

        n_samples = X_proc.shape[0]
        if n_samples == 0:
            raise ValueError("Empty input X")

        # 1) density estimation
        rho, distances, indices, used_sigma = self._estimate_density(X_proc)
        self.rho_ = rho
        self.metadata_["sigma"] = float(used_sigma)

        # 2) delta computation: distance to nearest point with higher density
        delta, nearest_higher = self._compute_delta(X_proc, rho, distances, indices)
        self.delta_ = delta
        self.neigh_idx_ = nearest_higher

        # 3) select centers via rho & delta thresholds
        # We can use combined score = rho * delta or use percentile thresholds separately.
        combined = rho * delta
        rho_thr = np.percentile(rho, self.rho_percentile)
        delta_thr = np.percentile(delta, self.delta_percentile)
        combined_thr = np.percentile(combined, self.rho_percentile)  # align with rho percentile

        self.metadata_.update({
            "rho_percentile": float(self.rho_percentile),
            "delta_percentile": float(self.delta_percentile),
            "rho_threshold": float(rho_thr),
            "delta_threshold": float(delta_thr),
            "combined_threshold": float(combined_thr),
        })

        # Candidate centers: points exceeding both rho and delta thresholds OR high combined
        center_mask = ((rho >= rho_thr) & (delta >= delta_thr)) | (combined >= combined_thr)
        center_indices = np.where(center_mask)[0]

        # If no centers found (rare), pick top-k combined values
        if center_indices.size == 0:
            top_k = max(1, int(np.ceil(n_samples * 0.01)))
            center_indices = np.argsort(-combined)[:top_k]

        self.centers_ = center_indices.tolist()

        # 4) assign each point to cluster of its nearest higher-density neighbor (propagate to center)
        labels = np.full(n_samples, -1, dtype=int)  # -1 denotes noise/unassigned
        # assign centers unique labels
        for cid, idx in enumerate(self.centers_):
            labels[idx] = cid

        # For others, follow nearest higher density chain until hitting a center
        for i in range(n_samples):
            if labels[i] != -1:
                continue  # already a center
            cur = i
            visited = set()
            while labels[cur] == -1:
                visited.add(cur)
                nxt = nearest_higher[cur]
                if nxt == -1:
                    # shouldn't happen: treat as noise
                    labels[cur] = -1
                    break
                if labels[nxt] != -1:
                    # reached a center or assigned point
                    labels[cur] = labels[nxt]
                    break
                if nxt in visited:
                    # loop detected; mark as noise
                    labels[cur] = -1
                    break
                cur = nxt

        # 5) Post-process clusters: remove small clusters -> mark as noise
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
        for cl, size in cluster_sizes.items():
            if size < self.min_cluster_size:
                labels[labels == cl] = -1  # mark all points in cluster as noise

        # Reindex cluster labels to be 0..n_clusters-1 (optional)
        assigned = np.unique(labels[labels != -1])
        remap = {old: new for new, old in enumerate(assigned.tolist())}
        new_labels = labels.copy()
        for old, new in remap.items():
            new_labels[labels == old] = new
        # centers list should be updated to only those that remain assigned
        surviving_centers = [c for c in self.centers_ if new_labels[c] != -1]
        # remap center ids to new labels indices
        center_label_map = {c: new_labels[c] for c in surviving_centers}

        self.labels_ = new_labels
        self.centers_ = surviving_centers
        self.metadata_.update({
            "n_samples": int(n_samples),
            "n_centers_initial": int(len(center_mask.nonzero()[0])),
            "n_centers_final": int(len(self.centers_)),
            "n_clusters_final": int(len(assigned)),
            "min_cluster_size": int(self.min_cluster_size),
        })

        # Save outputs to artifacts
        np.savetxt("artifacts/vdpc_labels.txt", self.labels_, fmt="%d")
        np.save("artifacts/vdpc_centers.npy", np.array(self.centers_, dtype=int))
        with open("artifacts/vdpc_metadata.json", "w") as fh:
            json.dump(self.metadata_, fh, indent=2)

        print("[INFO] VDPC clustering completed.")
        print(f"[INFO] Found {self.metadata_['n_clusters_final']} clusters (final).")
        print(f"[INFO] Centers (indices): {self.centers_}")

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# ----------------------------
# Example usage as script
# ----------------------------
if __name__ == "__main__":
    ensure_artifacts()
    # Load preprocessed features (train or full)
    if not os.path.exists("artifacts/X_train.csv"):
        raise FileNotFoundError("Preprocessed features not found at artifacts/X_train.csv. Run data_preprocessing.py first.")

    X = np.loadtxt("artifacts/X_train.csv", delimiter=",")
    print(f"[INFO] Loaded features shape: {X.shape}")

    vdpc = VDPC(
        k=20,
        sigma=None,
        rho_percentile=90.0,
        delta_percentile=90.0,
        min_cluster_size=8,
        use_scaler=True,
    )

    vdpc.fit(X)

    # metadata and labels are saved in artifacts/
