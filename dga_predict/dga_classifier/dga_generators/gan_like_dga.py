"""
GAN-like DGA Generator
Simulates Generative Adversarial Network behavior
Creates domains that are hard to distinguish from benign
"""
import random
import string
import hashlib
from datetime import datetime

# Distribution learned from benign domains (simulated GAN training)
BENIGN_DISTRIBUTION = {
    'char_freq': {
        'a': 0.08, 'e': 0.12, 'i': 0.07, 'o': 0.08, 'u': 0.03,
        't': 0.09, 'n': 0.07, 's': 0.06, 'r': 0.06, 'h': 0.06,
        'd': 0.04, 'l': 0.04, 'c': 0.03, 'm': 0.02, 'f': 0.02,
    },
    'bigram_freq': {
        'th': 0.04, 'he': 0.03, 'in': 0.02, 'er': 0.02, 'an': 0.02,
        're': 0.02, 'ed': 0.02, 'on': 0.02, 'es': 0.02, 'st': 0.02,
    },
    'length_dist': {8: 0.1, 9: 0.15, 10: 0.2, 11: 0.2, 12: 0.15, 13: 0.1, 14: 0.05, 15: 0.05}
}

def sample_from_distribution(dist):
    """Sample from probability distribution"""
    items = list(dist.keys())
    probs = list(dist.values())
    return random.choices(items, weights=probs, k=1)[0]


def generate_gan_like_domain(seed, add_tld=False):
    """
    Generate domain using GAN-like approach
    Samples from learned distribution of benign domains
    """
    random.seed(seed)
    
    # Sample length from distribution
    length = sample_from_distribution(BENIGN_DISTRIBUTION['length_dist'])
    
    domain = ""
    
    # Generate using character frequency distribution
    for i in range(length):
        if i == 0:
            # First character: prefer common starts
            char = random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
        elif i < length - 1:
            # Use bigram if possible
            if random.random() < 0.3:
                bigram = sample_from_distribution(BENIGN_DISTRIBUTION['bigram_freq'])
                if len(domain) + len(bigram) <= length:
                    domain += bigram
                    continue
            # Otherwise use character frequency
            char = sample_from_distribution(BENIGN_DISTRIBUTION['char_freq'])
        else:
            # Last character: prefer common endings
            char = random.choice(['a', 'e', 'i', 'o', 'u', 'r', 's', 't', 'n', 'd', 'l'])
        
        domain += char
    
    # Ensure length
    domain = domain[:length]
    while len(domain) < length:
        char = sample_from_distribution(BENIGN_DISTRIBUTION['char_freq'])
        domain += char
    
    # Sometimes add number (like real domains)
    if random.random() < 0.1:
        pos = random.randint(0, len(domain) - 1)
        domain = domain[:pos] + str(random.randint(0, 9)) + domain[pos+1:]
    
    if add_tld:
        tlds = ['com', 'net', 'org', 'info', 'co']
        domain += '.' + random.choice(tlds)
    
    return domain[:length] if not add_tld else domain[:length] + '.' + random.choice(tlds)


def generate_domains(num_domains, start_date=None, add_tld=False):
    """
    Generate GAN-like domains
    
    Args:
        num_domains: Số lượng domain
        start_date: Ngày bắt đầu
        add_tld: Có thêm TLD không
    """
    if start_date is None:
        start_date = datetime.now()
    
    domains = []
    base_seed = int(start_date.timestamp())
    
    for i in range(num_domains):
        seed = base_seed + i * 13  # Variation
        domain = generate_gan_like_domain(seed, add_tld)
        domains.append(domain)
    
    return domains

