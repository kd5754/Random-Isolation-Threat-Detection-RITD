import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# 1. Create artifacts directory if not exists
# ---------------------------------------------------------------
def ensure_artifacts():
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        print("[INFO] Created 'artifacts/' directory.")

# ---------------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------------
def load_dataset(path="dataset.csv"):
    """
    Loads raw dataset.

    Expected: A CSV containing 'label' column as the target.

    Returns:
        df: pandas DataFrame
    """
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)

    print(f"[INFO] Loaded dataset shape: {df.shape}")
    return df

# ---------------------------------------------------------------
# 3. Clean & Encode Features
# ---------------------------------------------------------------
def preprocess(df):
    """
    Cleans and preprocesses the dataset.

    Steps:
        1. Drop rows with missing target labels
        2. Fill missing features
        3. Label-encode categorical columns
        4. Standardize numerical columns

    Returns:
        X_scaled  – preprocessed features
        y_encoded – encoded target vector
    """

    print("[INFO] Starting preprocessing...")

    # Drop rows with missing label
    df = df.dropna(subset=["label"])

    # Handle missing values by filling with median
    df = df.fillna(df.median(numeric_only=True))

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Label encode categorical columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        print(f"[INFO] Encoded categorical column: {col}")

    # Split features/labels
    X = df.drop("label", axis=1).values
    y = df["label"].values

    print(f"[INFO] Feature dimension: {X.shape[1]}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("[INFO] Features scaled.")

    return X_scaled, y

# ---------------------------------------------------------------
# 4. Save preprocessed data
# ---------------------------------------------------------------
def save_artifacts(X_train, X_test, y_train, y_test):
    """
    Saves preprocessing result into artifacts/ folder.
    """
    np.savetxt("artifacts/X_train.csv", X_train, delimiter=",")
    np.savetxt("artifacts/X_test.csv", X_test, delimiter=",")
    np.savetxt("artifacts/y_train.csv", y_train, delimiter=",")
    np.savetxt("artifacts/y_test.csv", y_test, delimiter=",")

    print("[INFO] Saved all processed datasets into artifacts/")

# ---------------------------------------------------------------
# 5. Main Execution Flow
# ---------------------------------------------------------------
if __name__ == "__main__":
    ensure_artifacts()

    df = load_dataset("dataset.csv")
    X, y = preprocess(df)

    # Train/test split for experiments
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"[INFO] Train size: {X_train.shape}")
    print(f"[INFO] Test size : {X_test.shape}")

    save_artifacts(X_train, X_test, y_train, y_test)

    print("[INFO] Preprocessing Complete ✓")
