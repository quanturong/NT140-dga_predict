"""Train and test bigram classifier (supports fast mode with 1-fold)"""
import numpy as np
import dga_classifier.data as data
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn import feature_extraction, metrics
from sklearn.model_selection import StratifiedKFold, train_test_split


def build_model(max_features):
    """Simple logistic regression classifier using Keras"""
    model = Sequential()
    model.add(Dense(1, input_dim=max_features, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def run(max_epoch=50, nfolds=5, batch_size=128):
    """Run train/test on logistic regression model with K-fold or single split"""
    indata = data.get_data()
    X_raw = [x[1] for x in indata]
    y_raw = np.array([0 if x[0] == 'benign' else 1 for x in indata])

    print("Vectorizing data...")
    vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2, 2))
    X = vectorizer.fit_transform(X_raw)
    max_features = X.shape[1]

    final_data = []

    # --- FAST MODE ---
    if nfolds <= 1:
        print("\nRunning in FAST MODE (single train/test split)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, stratify=y_raw, random_state=42)
        folds = [(X_train, X_test, y_train, y_test)]
    else:
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
        folds = []
        for train_idx, test_idx in skf.split(X, y_raw):
            folds.append((X[train_idx], X[test_idx], y_raw[train_idx], y_raw[test_idx]))

    # --- TRAIN LOOP ---
    for fold, (X_train, X_test, y_train, y_test) in enumerate(folds, start=1):
        print(f"\nFold {fold}/{len(folds)}")
        model = build_model(max_features)
        best_auc, best_iter = 0.0, -1
        out_data = {}

        for ep in range(max_epoch):
            model.fit(np.asarray(X_train.todense()), y_train, batch_size=batch_size, epochs=1, verbose=0)
            preds = model.predict(np.asarray(X_test.todense()), verbose=0)
            t_auc = metrics.roc_auc_score(y_test, preds)
            print(f"Epoch {ep}: auc={t_auc:.6f} (best={best_auc:.6f})")

            if t_auc > best_auc:
                best_auc, best_iter = t_auc, ep
                out_data = {
                    "y": y_test,
                    "probs": preds,
                    "confusion_matrix": metrics.confusion_matrix(y_test, preds > 0.5)
                }
            elif ep - best_iter > 5:
                break

        print(out_data["confusion_matrix"])
        final_data.append(out_data)

    return final_data
