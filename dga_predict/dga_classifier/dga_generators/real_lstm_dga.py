"""
Real LSTM-based DGA Generator
Trains an actual LSTM model on benign domains to generate DGA domains
"""
import os
# Set CPU mode if CUDA has issues (must be before tensorflow import)
# Uncomment the line below to force CPU mode if you get CUDA errors:
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import random
import string
import pickle
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Try to configure GPU, fallback to CPU on error
try:
    import tensorflow as tf
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

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'lstm_generator_model.h5')
VOCAB_FILE = os.path.join(os.path.dirname(__file__), 'lstm_generator_vocab.pkl')

def build_char_vocab(domains):
    """Build character vocabulary from domains"""
    chars = set()
    for domain in domains:
        chars.update(domain.lower())
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    char_to_idx['<PAD>'] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def prepare_sequences(domains, char_to_idx, maxlen=20):
    """Convert domains to sequences"""
    sequences = []
    for domain in domains:
        seq = [char_to_idx.get(c, 0) for c in domain.lower()]
        sequences.append(seq)
    return pad_sequences(sequences, maxlen=maxlen, padding='pre')

def build_lstm_generator(vocab_size, embedding_dim=128, lstm_units=256, maxlen=20):
    """Build LSTM generator model"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim),  # Removed deprecated input_length
        LSTM(lstm_units, return_sequences=True),
        LSTM(lstm_units),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def train_lstm_generator(domains, epochs=100, batch_size=128):
    """Train LSTM generator on benign domains
    
    Note: Increased epochs to 100 for better learning of benign patterns
    """
    print(f"Training LSTM generator on {len(domains)} benign domains ({epochs} epochs)...")
    print("  (Using CPU if GPU unavailable - this may take longer)")
    
    # Build vocabulary
    char_to_idx, idx_to_char = build_char_vocab(domains)
    vocab_size = len(char_to_idx)
    maxlen = min(20, max(len(d) for d in domains))
    
    # Prepare sequences
    X = prepare_sequences(domains, char_to_idx, maxlen)
    
    # Create training data (predict next character)
    X_train, y_train = [], []
    for seq in X:
        for i in range(1, len(seq)):
            if seq[i] != 0:  # Not padding
                X_train.append(seq[:i])
                y_train.append(seq[i])
    
    if len(X_train) == 0:
        # Fallback: use simple patterns
        return None, None, None
    
    X_train = pad_sequences(X_train, maxlen=maxlen, padding='pre')
    y_train = np.array(y_train)
    
    # Build and train model
    model = build_lstm_generator(vocab_size, maxlen=maxlen)
    
    # Train
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Save
    model.save(MODEL_FILE)
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump((char_to_idx, idx_to_char, maxlen), f)
    
    print(f"✓ LSTM generator trained and saved")
    return model, char_to_idx, idx_to_char

def generate_domain_lstm(model, char_to_idx, idx_to_char, maxlen, seed=None, temperature=0.8, max_length=15):
    """Generate domain using trained LSTM"""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    
    domain = ""
    sequence = [0] * maxlen  # Start with padding
    
    for _ in range(max_length):
        # Predict next character
        input_seq = np.array([sequence])
        probs = model.predict(input_seq, verbose=0)[0]
        
        # Apply temperature for diversity
        probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)
        
        # Sample
        next_char_idx = np.random.choice(len(probs), p=probs)
        
        if next_char_idx == 0:  # Padding token
            break
        
        next_char = idx_to_char.get(next_char_idx, '')
        if next_char and next_char != '<PAD>':
            domain += next_char
            sequence = sequence[1:] + [next_char_idx]
        else:
            break
    
    return domain if domain else ''.join(random.choices(string.ascii_lowercase, k=8))

def generate_domains_with_benign(num_domains, benign_domains, start_date=None, add_tld=False, force_retrain=False):
    """
    Generate domains using real LSTM generator with provided benign domains
    
    Args:
        num_domains: Số lượng domain
        benign_domains: List of benign domains for training
        start_date: Ngày bắt đầu
        add_tld: Có thêm TLD không
        force_retrain: Force retrain model
    """
    if start_date is None:
        start_date = datetime.now()
    
    # Try to load existing model
    model = None
    char_to_idx = None
    idx_to_char = None
    maxlen = 20
    
    if not force_retrain and os.path.exists(MODEL_FILE) and os.path.exists(VOCAB_FILE):
        try:
            model = load_model(MODEL_FILE)
            with open(VOCAB_FILE, 'rb') as f:
                char_to_idx, idx_to_char, maxlen = pickle.load(f)
            print("✓ Loaded existing LSTM generator model")
        except:
            force_retrain = True
    
    # If no model, need to train
    if model is None or force_retrain:
        if len(benign_domains) < 1000:
            print("⚠ Not enough benign domains, using fallback generator")
            return generate_domains_fallback(num_domains, start_date, add_tld)
        
        # Use provided benign domains (limit to 50k for training speed)
        training_domains = benign_domains[:50000]
        model, char_to_idx, idx_to_char = train_lstm_generator(training_domains)
        if model is None:
            return generate_domains_fallback(num_domains, start_date, add_tld)
    
    # Generate domains
    domains = []
    base_seed = int(start_date.timestamp())
    
    for i in range(num_domains):
        seed = base_seed + i
        domain = generate_domain_lstm(model, char_to_idx, idx_to_char, maxlen, seed)
        
        # Ensure reasonable length
        if len(domain) < 5:
            domain += ''.join(random.choices(string.ascii_lowercase, k=5))
        domain = domain[:20]  # Max length
        
        if add_tld:
            tlds = ['com', 'net', 'org', 'info']
            domain += '.' + random.choice(tlds)
        
        domains.append(domain)
    
    return domains


def generate_domains(num_domains, start_date=None, add_tld=False, force_retrain=False):
    """
    Generate domains using real LSTM generator
    
    Args:
        num_domains: Số lượng domain
        start_date: Ngày bắt đầu
        add_tld: Có thêm TLD không
        force_retrain: Force retrain model
    """
    if start_date is None:
        start_date = datetime.now()
    
    # Try to load existing model
    model = None
    char_to_idx = None
    idx_to_char = None
    maxlen = 20
    
    if not force_retrain and os.path.exists(MODEL_FILE) and os.path.exists(VOCAB_FILE):
        try:
            model = load_model(MODEL_FILE)
            with open(VOCAB_FILE, 'rb') as f:
                char_to_idx, idx_to_char, maxlen = pickle.load(f)
            print("✓ Loaded existing LSTM generator model")
        except:
            force_retrain = True
    
    # If no model, need to train (requires benign domains)
    if model is None or force_retrain:
        # Get benign domains for training
        try:
            from dga_classifier import data
            data_list = data.get_data(force=False)
            benign_domains = [d[1] for d in data_list if d[0] == 'benign'][:50000]  # Use up to 50k
            
            if len(benign_domains) < 1000:
                # Not enough data, use fallback
                print("⚠ Not enough benign domains, using fallback generator")
                return generate_domains_fallback(num_domains, start_date, add_tld)
            
            model, char_to_idx, idx_to_char = train_lstm_generator(benign_domains)
            if model is None:
                return generate_domains_fallback(num_domains, start_date, add_tld)
        except Exception as e:
            print(f"⚠ Error training LSTM: {e}, using fallback")
            return generate_domains_fallback(num_domains, start_date, add_tld)
    
    # Generate domains
    domains = []
    base_seed = int(start_date.timestamp())
    
    for i in range(num_domains):
        seed = base_seed + i
        domain = generate_domain_lstm(model, char_to_idx, idx_to_char, maxlen, seed)
        
        # Ensure reasonable length
        if len(domain) < 5:
            domain += ''.join(random.choices(string.ascii_lowercase, k=5))
        domain = domain[:20]  # Max length
        
        if add_tld:
            tlds = ['com', 'net', 'org', 'info']
            domain += '.' + random.choice(tlds)
        
        domains.append(domain)
    
    return domains

def generate_domains_fallback(num_domains, start_date, add_tld):
    """Fallback generator if LSTM training fails"""
    domains = []
    for i in range(num_domains):
        length = random.randint(8, 15)
        domain = ''.join(random.choices(string.ascii_lowercase, k=length))
        if add_tld:
            domain += '.' + random.choice(['com', 'net', 'org'])
        domains.append(domain)
    return domains

