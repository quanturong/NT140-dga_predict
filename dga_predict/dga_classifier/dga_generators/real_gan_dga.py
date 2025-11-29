"""
Real GAN-based DGA Generator
Uses actual Generative Adversarial Network (Generator + Discriminator)
"""
import numpy as np
import random
import string
import os
import pickle
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Input, Lambda, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

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
    model = Sequential([
        Embedding(vocab_size, 128, input_length=maxlen),
        LSTM(128),
        Dense(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_gan(domains, epochs=50, batch_size=128):
    """Train GAN on benign domains"""
    print("Training GAN generator on benign domains...")
    
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
        generated_probs = generated_probs.reshape((batch_size, maxlen, vocab_size))
        sequences = []
        for i in range(batch_size):
            seq = []
            for j in range(maxlen):
                char_idx = np.random.choice(vocab_size, p=generated_probs[i, j])
                seq.append(char_idx)
            sequences.append(seq)
        return np.array(sequences)
    
    # Combined model (generator + discriminator)
    z = Input(shape=(latent_dim,))
    generated_probs = generator(z)
    # Reshape to (batch, maxlen, vocab_size) and sample using argmax
    generated_reshaped = Reshape((maxlen, vocab_size))(generated_probs)
    generated_seqs = Lambda(lambda x: tf.argmax(x, axis=-1))(generated_reshaped)
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
        gen_seqs = sample_sequences(gen_probs, batch_size, maxlen, vocab_size)
        
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
        
        # Apply temperature
        char_probs = np.log(char_probs + 1e-10) / temperature
        char_probs = np.exp(char_probs)
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

