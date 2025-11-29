"""
Adversarial-trained DGA Generator
Trains generator to bypass detection models using adversarial training
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
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Input
from tensorflow.keras.optimizers import Adam
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

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'adversarial_generator_model.h5')
VOCAB_FILE = os.path.join(os.path.dirname(__file__), 'adversarial_generator_vocab.pkl')

def build_char_vocab(domains):
    """Build character vocabulary"""
    chars = set()
    for domain in domains:
        chars.update(domain.lower())
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    char_to_idx['<PAD>'] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def build_adversarial_generator(vocab_size, maxlen=20, latent_dim=100):
    """Build generator optimized to bypass detection"""
    model = Sequential([
        Dense(256, input_dim=latent_dim, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dense(maxlen * vocab_size),
        Dense(maxlen * vocab_size, activation='softmax')
    ])
    return model

def build_detector_model(vocab_size, maxlen=20):
    """Build detector model (simulates detection system)"""
    model = Sequential([
        Embedding(vocab_size, 128),  # Removed deprecated input_length
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # 0 = benign, 1 = DGA
    ])
    return model

def train_adversarial_generator(domains, epochs=50, batch_size=128):
    """Train adversarial generator to bypass detector
    
    Note: Increased epochs to 50 for better adversarial training
    """
    print(f"Training adversarial generator on {len(domains)} benign domains ({epochs} epochs)...")
    print("  (Using CPU if GPU unavailable - this may take longer)")
    
    # Build vocabulary
    char_to_idx, idx_to_char = build_char_vocab(domains)
    vocab_size = len(char_to_idx)
    maxlen = min(20, max(len(d) for d in domains))
    
    # Prepare data
    X_real = []
    for domain in domains[:5000]:
        seq = [char_to_idx.get(c, 0) for c in domain.lower()[:maxlen]]
        X_real.append(seq)
    X_real = pad_sequences(X_real, maxlen=maxlen, padding='pre')
    
    if len(X_real) < 100:
        return None, None, None
    
    # Build models
    latent_dim = 100
    generator = build_adversarial_generator(vocab_size, maxlen, latent_dim)
    detector = build_detector_model(vocab_size, maxlen)
    
    # Train detector first on real data
    y_real = np.zeros((len(X_real), 1))  # 0 = benign
    detector.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    detector.fit(X_real, y_real, epochs=10, batch_size=batch_size, verbose=0)
    
    # Adversarial training: generator tries to fool detector
    detector.trainable = False
    
    z = Input(shape=(latent_dim,))
    generated = generator(z)
    validity = detector(generated)
    adversarial = Model(z, validity)
    adversarial.compile(optimizer=Adam(0.0002), loss='binary_crossentropy')
    
    # Train generator to minimize detector output (make it think generated = benign)
    target_benign = np.zeros((batch_size, 1))  # Target: detector thinks it's benign
    
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = adversarial.train_on_batch(noise, target_benign)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: G_loss={g_loss:.4f}")
    
    # Save
    generator.save(MODEL_FILE)
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump((char_to_idx, idx_to_char, maxlen, latent_dim), f)
    
    print(f"✓ Adversarial generator trained and saved")
    return generator, char_to_idx, idx_to_char

def generate_domain_adversarial(generator, char_to_idx, idx_to_char, maxlen, latent_dim, seed=None, temperature=0.8):
    """Generate domain using adversarial generator"""
    if seed:
        np.random.seed(seed)
    
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated = generator.predict(noise, verbose=0)[0]
    
    # Reshape to (maxlen, vocab_size)
    vocab_size = len(char_to_idx)
    generated = generated.reshape((maxlen, vocab_size))
    
    domain = ""
    for i in range(maxlen):
        char_probs = generated[i]
        
        # Apply temperature
        char_probs = np.log(char_probs + 1e-10) / temperature
        char_probs = np.exp(char_probs)
        char_probs = char_probs / np.sum(char_probs)
        
        # Sample
        char_idx = np.random.choice(vocab_size, p=char_probs)
        
        if char_idx == 0:
            break
        
        char = idx_to_char.get(char_idx, '')
        if char and char != '<PAD>':
            domain += char
        else:
            break
    
    return domain if domain else ''.join(random.choices(string.ascii_lowercase, k=8))

def generate_domains_with_benign(num_domains, benign_domains, start_date=None, add_tld=False, force_retrain=False):
    """
    Generate domains using adversarial-trained generator with provided benign domains
    """
    if start_date is None:
        start_date = datetime.now()
    
    # Try to load existing model
    generator = None
    char_to_idx = None
    idx_to_char = None
    maxlen = 20
    latent_dim = 100
    
    if not force_retrain and os.path.exists(MODEL_FILE) and os.path.exists(VOCAB_FILE):
        try:
            generator = load_model(MODEL_FILE)
            with open(VOCAB_FILE, 'rb') as f:
                char_to_idx, idx_to_char, maxlen, latent_dim = pickle.load(f)
            print("✓ Loaded existing adversarial generator model")
        except:
            force_retrain = True
    
    # If no model, need to train
    if generator is None or force_retrain:
        if len(benign_domains) < 1000:
            return generate_domains_fallback(num_domains, start_date, add_tld)
        
        # Use provided benign domains (limit to 10k for training speed)
        training_domains = benign_domains[:10000]
        generator, char_to_idx, idx_to_char = train_adversarial_generator(training_domains)
        if generator is None:
            return generate_domains_fallback(num_domains, start_date, add_tld)
    
    # Generate domains
    domains = []
    base_seed = int(start_date.timestamp())
    
    for i in range(num_domains):
        seed = base_seed + i
        domain = generate_domain_adversarial(generator, char_to_idx, idx_to_char, maxlen, latent_dim, seed)
        
        if len(domain) < 5:
            domain += ''.join(random.choices(string.ascii_lowercase, k=5))
        domain = domain[:20]
        
        if add_tld:
            tlds = ['com', 'net', 'org', 'info']
            domain += '.' + random.choice(tlds)
        
        domains.append(domain)
    
    return domains


def generate_domains(num_domains, start_date=None, add_tld=False, force_retrain=False):
    """
    Generate domains using adversarial-trained generator
    """
    if start_date is None:
        start_date = datetime.now()
    
    # Try to load existing model
    generator = None
    char_to_idx = None
    idx_to_char = None
    maxlen = 20
    latent_dim = 100
    
    if not force_retrain and os.path.exists(MODEL_FILE) and os.path.exists(VOCAB_FILE):
        try:
            generator = load_model(MODEL_FILE)
            with open(VOCAB_FILE, 'rb') as f:
                char_to_idx, idx_to_char, maxlen, latent_dim = pickle.load(f)
            print("✓ Loaded existing adversarial generator model")
        except:
            force_retrain = True
    
    # If no model, need to train
    if generator is None or force_retrain:
        try:
            from dga_classifier import data
            data_list = data.get_data(force=False)
            benign_domains = [d[1] for d in data_list if d[0] == 'benign'][:10000]
            
            if len(benign_domains) < 1000:
                return generate_domains_fallback(num_domains, start_date, add_tld)
            
            generator, char_to_idx, idx_to_char = train_adversarial_generator(benign_domains)
            if generator is None:
                return generate_domains_fallback(num_domains, start_date, add_tld)
        except Exception as e:
            print(f"⚠ Error training adversarial generator: {e}, using fallback")
            return generate_domains_fallback(num_domains, start_date, add_tld)
    
    # Generate domains
    domains = []
    base_seed = int(start_date.timestamp())
    
    for i in range(num_domains):
        seed = base_seed + i
        domain = generate_domain_adversarial(generator, char_to_idx, idx_to_char, maxlen, latent_dim, seed)
        
        if len(domain) < 5:
            domain += ''.join(random.choices(string.ascii_lowercase, k=5))
        domain = domain[:20]
        
        if add_tld:
            tlds = ['com', 'net', 'org', 'info']
            domain += '.' + random.choice(tlds)
        
        domains.append(domain)
    
    return domains

def generate_domains_fallback(num_domains, start_date, add_tld):
    """Fallback generator"""
    domains = []
    for i in range(num_domains):
        length = random.randint(8, 15)
        domain = ''.join(random.choices(string.ascii_lowercase, k=length))
        if add_tld:
            domain += '.' + random.choice(['com', 'net', 'org'])
        domains.append(domain)
    return domains

