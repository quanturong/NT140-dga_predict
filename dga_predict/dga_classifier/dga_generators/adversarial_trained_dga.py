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
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Input, Lambda, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

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
    # Use Input layer to specify input shape explicitly
    input_layer = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, 128)(input_layer)
    lstm1 = LSTM(128, return_sequences=True)(embedding)
    lstm2 = LSTM(64)(lstm1)
    dropout = Dropout(0.5)(lstm2)
    dense1 = Dense(32, activation='relu')(dropout)
    output = Dense(1, activation='sigmoid')(dense1)  # 0 = benign, 1 = DGA
    
    model = Model(inputs=input_layer, outputs=output)
    return model

def train_adversarial_generator(domains, epochs=15, batch_size=128):
    """Train adversarial generator to bypass detector
    
    Note: Reduced epochs to 15 for faster training
    """
    print(f"Training adversarial generator on {len(domains)} benign domains ({epochs} epochs)...")
    print("  (Using CPU if GPU unavailable - this may take longer)")
    
    # Build vocabulary
    char_to_idx, idx_to_char = build_char_vocab(domains)
    vocab_size = len(char_to_idx)
    maxlen = min(20, max(len(d) for d in domains))
    
    # Prepare data
    X_real = []
    for domain in domains[:2000]:
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
    detector.fit(X_real, y_real, epochs=5, batch_size=batch_size, verbose=0)
    
    # Adversarial training: generator tries to fool detector
    detector.trainable = False
    
    # Generator optimizer (separate from combined model)
    generator_optimizer = Adam(0.0002)
    
    # Train generator to minimize detector output (make it think generated = benign)
    target_benign = np.zeros((batch_size, 1))  # Target: detector thinks it's benign
    
    for epoch in range(epochs):
        noise = tf.random.normal((batch_size, latent_dim))
        
        with tf.GradientTape() as gen_tape:
            # Generate probabilities
            gen_probs = generator(noise, training=True)
            gen_probs_reshaped = tf.reshape(gen_probs, (batch_size, maxlen, vocab_size))
            gen_probs_reshaped = tf.nn.softmax(gen_probs_reshaped, axis=-1)  # Normalize
            
            # Sample sequences and compute log probabilities (differentiable)
            gen_indices_list = []
            log_probs_list = []
            for i in range(batch_size):
                seq_indices = []
                seq_log_probs = []
                for j in range(maxlen):
                    probs_j = gen_probs_reshaped[i, j, :]
                    # Sample (non-differentiable, but we'll use log_probs for gradient)
                    sampled_idx = tf.random.categorical(tf.expand_dims(tf.math.log(probs_j + 1e-10), 0), 1)[0, 0]
                    seq_indices.append(sampled_idx)
                    # Log probability (differentiable w.r.t. probs_j)
                    log_prob = tf.math.log(probs_j[sampled_idx] + 1e-10)
                    seq_log_probs.append(log_prob)
                gen_indices_list.append(seq_indices)
                log_probs_list.append(tf.reduce_sum(seq_log_probs))
            
            gen_indices = tf.cast(tf.stack([tf.stack(seq) for seq in gen_indices_list]), tf.int32)
            gen_log_probs = tf.stack(log_probs_list)  # (batch_size,)
            
            # Get detector prediction (non-differentiable path)
            disc_pred = detector(gen_indices, training=False)
            rewards = tf.squeeze(disc_pred)  # (batch_size,)
            
            # Adversarial loss: we want detector to output 0 (benign)
            # So we want to minimize detector output
            # REINFORCE: loss = -mean(log_prob * (1 - reward))
            # Higher reward (closer to 0 = benign) is better, so use (1 - reward) as reward signal
            gen_loss = -tf.reduce_mean(gen_log_probs * (1.0 - rewards))
        
        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_gradients = [g for g in gen_gradients if g is not None]
        if len(gen_gradients) > 0:
            generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            g_loss = gen_loss.numpy()
        else:
            g_loss = 0.0
            print("⚠ No gradients for generator, skipping update")
        
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
        
        # Ensure non-negative and handle NaN/Inf
        char_probs = np.maximum(char_probs, 0)
        char_probs = np.nan_to_num(char_probs, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Apply temperature
        char_probs = np.log(char_probs + 1e-10) / temperature
        char_probs = np.exp(char_probs)
        
        # Normalize to ensure sum = 1
        probs_sum = np.sum(char_probs)
        if probs_sum > 0:
            char_probs = char_probs / probs_sum
        else:
            # Fallback to uniform if all zeros
            char_probs = np.ones(vocab_size) / vocab_size
        
        # Final normalization to fix floating point errors
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
        
        # Use provided benign domains (limit to 5k for training speed)
        training_domains = benign_domains[:5000]
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

