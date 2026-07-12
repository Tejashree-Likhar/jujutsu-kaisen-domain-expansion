"""
Trains a classifier on the landmark data collected by data_collection.py
and saves it to models/domain_classifier.pkl

Usage:
    python src/train_model.py
"""

import os
import sys
import glob
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.domains import ALL_LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")


def load_dataset():
    X, y = [], []
    missing = []
    for label in ALL_LABELS:
        path = os.path.join(DATA_DIR, f"{label}.npy")
        if not os.path.exists(path):
            missing.append(label)
            continue
        arr = np.load(path)
        X.append(arr)
        y.extend([label] * arr.shape[0])

    if missing:
        print("WARNING: no data found for:", missing)
        print("Run data_collection.py for these labels before training.")

    if not X:
        raise SystemExit("No training data found at all. Run data_collection.py first.")

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    return X, y


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_dataset()
    print(f"Loaded {X.shape[0]} samples across {len(set(y))} classes.")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Evaluation on held-out test split ---")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # retrain on ALL data before saving, for max use of your collected samples
    clf_final = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    clf_final.fit(X, y_enc)

    bundle = {"model": clf_final, "label_encoder": encoder}
    out_path = os.path.join(MODEL_DIR, "domain_classifier.pkl")
    joblib.dump(bundle, out_path)
    print(f"\nSaved trained model -> {out_path}")
    print("Now run: python src/main.py")


if __name__ == "__main__":
    main()
