"""
Real GAN-based DGA Generator
Uses actual Generative Adversarial Network (Generator + Discriminator)
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

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'gan_generator_model.h5')
VOCAB_FILE = os.path.join(os.path.dirname(__file__), 'gan_generator_vocab.pkl')

def build_char_vocab(domains):
    """Build character vocabulary"""
    chars = set()
    for domain in domains:
        chars.update(domain.lower())
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    char_to_idx['<PAD>'] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def build_generator(vocab_size, maxlen=20, latent_dim=100):
    """Build GAN Generator - outputs sequences directly"""
    model = Sequential([
        Dense(256, input_dim=latent_dim, activation='relu'),
        Dense(512, activation='relu'),
        Dense(maxlen * 128, activation='relu'),
        Dense(maxlen * vocab_size, activation='softmax')
    ])
    return model

def build_discriminator(vocab_size, maxlen=20):
    """Build GAN Discriminator"""
    # Use Input layer to specify input shape explicitly
    from tensorflow.keras.layers import Input
    input_layer = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, 128)(input_layer)
    lstm = LSTM(128)(embedding)
    dense1 = Dense(64)(lstm)
    dropout = Dropout(0.5)(dense1)
    output = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

def train_gan(domains, epochs=100, batch_size=128):
    """Train GAN on benign domains
    
    Note: Increased epochs to 100 for better adversarial training
    """
    print(f"Training GAN generator on {len(domains)} benign domains ({epochs} epochs)...")
    print("  (Using CPU if GPU unavailable - this may take longer)")
    
    # Build vocabulary
    char_to_idx, idx_to_char = build_char_vocab(domains)
    vocab_size = len(char_to_idx)
    maxlen = min(20, max(len(d) for d in domains))
    
    # Prepare real data
    X_real = []
    for domain in domains[:10000]:  # Limit for training speed
        seq = [char_to_idx.get(c, 0) for c in domain.lower()[:maxlen]]
        X_real.append(seq)
    X_real = pad_sequences(X_real, maxlen=maxlen, padding='pre')
    
    if len(X_real) < 100:
        return None, None, None
    
    # Build models
    latent_dim = 100
    generator = build_generator(vocab_size, maxlen, latent_dim)
    discriminator = build_discriminator(vocab_size, maxlen)
    
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Helper function to sample sequences from generator output
    def sample_sequences(generated_probs, batch_size, maxlen, vocab_size):
        """Sample sequences from probability distribution"""
        # Ensure input is numpy array
        if not isinstance(generated_probs, np.ndarray):
            generated_probs = np.array(generated_probs)
        
        # Flatten and reshape to ensure correct shape
        total_size = batch_size * maxlen * vocab_size
        if generated_probs.size != total_size:
            # Try to reshape to expected size
            generated_probs = generated_probs.flatten()[:total_size]
            if generated_probs.size < total_size:
                # Pad with zeros if needed
                padding = np.zeros(total_size - generated_probs.size)
                generated_probs = np.concatenate([generated_probs, padding])
        
        # Reshape to (batch_size, maxlen, vocab_size)
        generated_probs = generated_probs.reshape((batch_size, maxlen, vocab_size))
        
        # Validate shape
        if generated_probs.shape != (batch_size, maxlen, vocab_size):
            raise ValueError(f"Invalid shape after reshape: {generated_probs.shape}, expected ({batch_size}, {maxlen}, {vocab_size})")
        
        sequences = []
        for i in range(batch_size):
            seq = []
            for j in range(maxlen):
                # Get probability distribution for this position
                probs = generated_probs[i, j].copy()
                
                # Normalize probabilities to ensure they sum to 1
                # Handle negative values and ensure non-negative
                probs = np.maximum(probs, 0)  # Remove negative values
                probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                probs_sum = np.sum(probs)
                
                if probs_sum > 0:
                    probs = probs / probs_sum  # Normalize
                else:
                    # If all zeros, use uniform distribution
                    probs = np.ones(vocab_size) / vocab_size
                
                # Ensure probabilities sum to 1 (fix floating point errors)
                probs = probs / np.sum(probs)
                
                # Sample from distribution
                try:
                    char_idx = np.random.choice(vocab_size, p=probs)
                except ValueError as e:
                    # Fallback if still has issues
                    char_idx = np.random.randint(0, vocab_size)
                
                seq.append(int(char_idx))
            sequences.append(seq)
        
        result = np.array(sequences, dtype=np.int32)
        
        # Validate output shape
        if result.shape != (batch_size, maxlen):
            raise ValueError(f"Invalid output shape: {result.shape}, expected ({batch_size}, {maxlen})")
        
        return result
    
    # Combined model (generator + discriminator)
    z = Input(shape=(latent_dim,))
    generated_probs = generator(z)
    # Reshape to (batch, maxlen, vocab_size) and sample using argmax
    generated_reshaped = Reshape((maxlen, vocab_size))(generated_probs)
    # Use argmax to get most likely character at each position
    # argmax on axis=-1 reduces (batch, maxlen, vocab_size) -> (batch, maxlen)
    generated_seqs = Lambda(lambda x: tf.argmax(x, axis=-1, output_type=tf.int32))(generated_reshaped)
    validity = discriminator(generated_seqs)
    
    discriminator.trainable = False
    combined = Model(z, validity)
    combined.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
    
    # Train
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Train Discriminator
        discriminator.trainable = True
        idx = np.random.randint(0, X_real.shape[0], batch_size)
        real_seqs = X_real[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_probs = generator.predict(noise, verbose=0)
        
        # Validate generator output shape
        expected_size = batch_size * maxlen * vocab_size
        if gen_probs.size != expected_size:
            # Try to fix shape
            gen_probs = gen_probs.flatten()[:expected_size]
            if gen_probs.size < expected_size:
                gen_probs = np.pad(gen_probs, (0, expected_size - gen_probs.size), mode='constant')
            gen_probs = gen_probs.reshape((batch_size, maxlen, vocab_size))
            # Re-flatten for sample_sequences
            gen_probs = gen_probs.flatten()
        
        gen_seqs = sample_sequences(gen_probs, batch_size, maxlen, vocab_size)
        
        # Validate sequences before training
        if gen_seqs.shape[1] == 0:
            raise ValueError(f"Generated sequences have zero length! Shape: {gen_seqs.shape}")
        
        # Ensure sequences are padded to maxlen for discriminator
        if gen_seqs.shape[1] < maxlen:
            # Pad sequences
            padding = np.zeros((batch_size, maxlen - gen_seqs.shape[1]), dtype=np.int32)
            gen_seqs = np.concatenate([gen_seqs, padding], axis=1)
        elif gen_seqs.shape[1] > maxlen:
            # Truncate sequences
            gen_seqs = gen_seqs[:, :maxlen]
        
        d_loss_real = discriminator.train_on_batch(real_seqs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_seqs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: D_loss={d_loss[0]:.4f}, G_loss={g_loss:.4f}")
    
    # Save
    generator.save(MODEL_FILE)
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump((char_to_idx, idx_to_char, maxlen, latent_dim), f)
    
    print(f"✓ GAN generator trained and saved")
    return generator, char_to_idx, idx_to_char

def generate_domain_gan(generator, char_to_idx, idx_to_char, maxlen, latent_dim, seed=None, temperature=0.8):
    """Generate domain using trained GAN"""
    if seed:
        np.random.seed(seed)
    
    # Generate from latent space
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated = generator.predict(noise, verbose=0)[0]
    
    # Reshape to (maxlen, vocab_size)
    vocab_size = len(char_to_idx)
    generated = generated.reshape((maxlen, vocab_size))
    
    # Convert to domain with temperature sampling
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
        
        if char_idx == 0:  # Padding
            break
        
        char = idx_to_char.get(char_idx, '')
        if char and char != '<PAD>':
            domain += char
        else:
            break
    
    return domain if domain else ''.join(random.choices(string.ascii_lowercase, k=8))

def generate_domains_with_benign(num_domains, benign_domains, start_date=None, add_tld=False, force_retrain=False):
    """
    Generate domains using real GAN generator with provided benign domains
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
            print("✓ Loaded existing GAN generator model")
        except:
            force_retrain = True
    
    # If no model, need to train
    if generator is None or force_retrain:
        if len(benign_domains) < 1000:
            return generate_domains_fallback(num_domains, start_date, add_tld)
        
        # Use provided benign domains (limit to 20k for training speed)
        training_domains = benign_domains[:20000]
        generator, char_to_idx, idx_to_char = train_gan(training_domains)
        if generator is None:
            return generate_domains_fallback(num_domains, start_date, add_tld)
    
    # Generate domains
    domains = []
    base_seed = int(start_date.timestamp())
    
    for i in range(num_domains):
        seed = base_seed + i
        domain = generate_domain_gan(generator, char_to_idx, idx_to_char, maxlen, latent_dim, seed)
        
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
    Generate domains using real GAN generator
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
            print("✓ Loaded existing GAN generator model")
        except:
            force_retrain = True
    
    # If no model, need to train
    if generator is None or force_retrain:
        try:
            from dga_classifier import data
            data_list = data.get_data(force=False)
            benign_domains = [d[1] for d in data_list if d[0] == 'benign'][:20000]
            
            if len(benign_domains) < 1000:
                return generate_domains_fallback(num_domains, start_date, add_tld)
            
            generator, char_to_idx, idx_to_char = train_gan(benign_domains)
            if generator is None:
                return generate_domains_fallback(num_domains, start_date, add_tld)
        except Exception as e:
            print(f"⚠ Error training GAN: {e}, using fallback")
            return generate_domains_fallback(num_domains, start_date, add_tld)
    
    # Generate domains
    domains = []
    base_seed = int(start_date.timestamp())
    
    for i in range(num_domains):
        seed = base_seed + i
        domain = generate_domain_gan(generator, char_to_idx, idx_to_char, maxlen, latent_dim, seed)
        
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

