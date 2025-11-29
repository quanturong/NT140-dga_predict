"""Manual statistical features + Random Forest classifier"""
import numpy as np
import pandas as pd
import dga_classifier.data as data
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from math import log2

def shannon_entropy(s):
    """Compute Shannon entropy"""
    if not s: return 0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * log2(p) for p in probs)

def extract_features(domains):
    """Extract statistical features from domain names"""
    vowels = set("aeiou")
    features = []
    common_words = ["bank","mail","shop","home","data","tech","news","info"]
    
    for d in domains:
        length = len(d)
        digits = sum(c.isdigit() for c in d)
        vowels_count = sum(c in vowels for c in d)
        entropy = shannon_entropy(d)
        vowel_ratio = vowels_count / length if length > 0 else 0
        digit_ratio = digits / length if length > 0 else 0
        has_word = any(w in d for w in common_words)
        features.append([
            length,
            digit_ratio,
            vowel_ratio,
            entropy,
            1 if has_word else 0
        ])
    return np.array(features)

def run(nfolds=10, n_estimators=100):
    """Train and evaluate Random Forest with manual features using cross-validation"""
    indata = data.get_data()
    X = [x[1] for x in indata]
    y_labels = np.array([0 if x[0] == "benign" else 1 for x in indata])

    feats = extract_features(X)

    final_data = []

    # --- FAST MODE ---
    if nfolds <= 1:
        print("\nRunning in FAST MODE (single train/test split)...")
        X_train, X_test, y_train, y_test = train_test_split(
            feats, y_labels, test_size=0.2, stratify=y_labels, random_state=42
        )
        folds = [(X_train, X_test, y_train, y_test)]
    else:
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
        folds = []
        for train_idx, test_idx in skf.split(feats, y_labels):
            folds.append((feats[train_idx], feats[test_idx], y_labels[train_idx], y_labels[test_idx]))

    # --- TRAIN LOOP ---
    for fold, (X_train, X_test, y_train, y_test) in enumerate(folds, start=1):
        print(f"\nFold {fold}/{len(folds)}")
        print("Training RandomForest...")
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, probs > 0.5)

        print(f"✓ AUC = {auc:.4f}")
        print(cm)

        final_data.append({
            "y": y_test,
            "probs": probs,
            "auc": auc,
            "confusion_matrix": cm
        })

    return final_data
