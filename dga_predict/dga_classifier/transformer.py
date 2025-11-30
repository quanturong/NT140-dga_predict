"""Train and test Transformer-based classifier for DGA detection"""
import os
# Set CPU mode if CUDA has issues (must be before tensorflow import)
# Uncomment the line below to force CPU mode if you get CUDA errors:
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import dga_classifier.data as data
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add, Lambda
)
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf

# Try to configure GPU, fallback to CPU on error
try:
    # Try to detect GPU
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        # GPU available, set memory growth to avoid allocation errors
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            # If GPU config fails, fallback to CPU
            print(f"⚠ GPU configuration failed: {e}, using CPU")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        print("ℹ No GPU detected, using CPU")
except Exception as e:
    # If tensorflow not available or error, use CPU
    print(f"⚠ TensorFlow GPU setup failed: {e}, using CPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer for Transformer"""
    def __init__(self, maxlen, d_model, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.d_model = d_model
        
        # Create positional encoding
        position = np.arange(maxlen)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((maxlen, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        # Store as a constant tensor
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        # inputs shape: (batch_size, maxlen, d_model)
        # pos_encoding shape: (maxlen, d_model)
        # Broadcast: (1, maxlen, d_model)
        pos_encoding = tf.expand_dims(self.pos_encoding, 0)
        return inputs + pos_encoding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'maxlen': self.maxlen,
            'd_model': self.d_model
        })
        return config


def transformer_encoder_block(inputs, d_model, num_heads, ff_dim, dropout_rate=0.1):
    """Single Transformer encoder block"""
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=d_model // num_heads,
        dropout=dropout_rate
    )(inputs, inputs)
    
    # Add & Norm
    attention_output = Dropout(dropout_rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([inputs, attention_output]))
    
    # Feed-forward network
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    
    # Add & Norm
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ffn_output]))
    
    return out2


def build_model(max_features, maxlen, d_model=128, num_heads=8, num_layers=2, ff_dim=512, dropout_rate=0.1):
    """Build Transformer-based classifier"""
    # Input
    inputs = Input(shape=(maxlen,))
    
    # Embedding
    embedding = Embedding(max_features, d_model)(inputs)
    
    # Add positional encoding using custom layer
    pos_encoding_layer = PositionalEncoding(maxlen, d_model)
    x = pos_encoding_layer(embedding)
    
    # Transformer encoder blocks
    for _ in range(num_layers):
        x = transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Classification head
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model


def run(max_epoch=20, nfolds=5, batch_size=128):
    """Run Transformer classifier with cross-validation"""
    indata = data.get_data()
    X_raw = [x[1] for x in indata]
    y_raw = np.array([0 if x[0] == 'benign' else 1 for x in indata])

    # Build character vocabulary
    valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(X_raw)))}
    max_features = len(valid_chars) + 1
    maxlen = np.max([len(x) for x in X_raw])

    X = [[valid_chars[c] for c in s] for s in X_raw]
    X = sequence.pad_sequences(X, maxlen=maxlen)

    final_data = []

    # --- FAST MODE ---
    if nfolds <= 1:
        print("\nRunning in FAST MODE (single train/test split)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_raw, test_size=0.2, stratify=y_raw, random_state=42
        )
        folds = [(X_train, X_test, y_train, y_test)]
    else:
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
        folds = []
        for train_idx, test_idx in skf.split(X, y_raw):
            folds.append((X[train_idx], X[test_idx], y_raw[train_idx], y_raw[test_idx]))

    # --- TRAIN LOOP ---
    for fold, (X_train, X_test, y_train, y_test) in enumerate(folds, start=1):
        print(f"\nFold {fold}/{len(folds)}")
        model = build_model(max_features, maxlen)
        best_auc, best_iter = 0.0, -1
        out_data = {}

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
            preds = model.predict(X_test, verbose=0)
            t_auc = metrics.roc_auc_score(y_test, preds)
            print(f"Epoch {ep}: auc={t_auc:.6f} (best={best_auc:.6f})")

            if t_auc > best_auc:
                best_auc, best_iter = t_auc, ep
                out_data = {
                    "y": y_test,
                    "probs": preds,
                    "confusion_matrix": metrics.confusion_matrix(y_test, preds > 0.5)
                }
            elif ep - best_iter > 3:
                break

        print(out_data["confusion_matrix"])
        final_data.append(out_data)

    return final_data

