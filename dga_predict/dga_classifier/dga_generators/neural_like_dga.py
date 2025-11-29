"""
Neural-like DGA
Sử dụng pattern phức tạp giống như neural network tạo ra
Kết hợp n-gram patterns với randomness
"""
import random
import string
from datetime import datetime

# Common character transitions (learned from benign domains)
COMMON_BIGRAMS = [
    'th', 'he', 'in', 'er', 'an', 're', 'ed', 'on', 'es', 'st',
    'en', 'at', 'to', 'nt', 'nd', 'ou', 'is', 'it', 'or', 'ti',
    'as', 'to', 'be', 'of', 'and', 'ar', 'al', 'le', 'co', 'de'
]

COMMON_TRIGRAMS = [
    'the', 'and', 'ing', 'ion', 'tio', 'ent', 'ati', 'for', 'her',
    'ter', 'hat', 'tha', 'ere', 'ate', 'his', 'con', 'res', 'ver'
]

def generate_neural_like_domain(length=12, seed=None, add_tld=False):
    """
    Generate domain với pattern giống neural network
    Kết hợp common n-grams với randomness
    """
    if seed:
        random.seed(seed)
    
    domain = ""
    
    # Strategy: mix common patterns với random
    strategy = random.choice(['bigram_heavy', 'trigram_heavy', 'mixed', 'random'])
    
    if strategy == 'bigram_heavy':
        # Sử dụng nhiều bigrams
        while len(domain) < length:
            if len(domain) < length - 1:
                bigram = random.choice(COMMON_BIGRAMS)
                domain += bigram
            else:
                domain += random.choice(string.ascii_lowercase)
    
    elif strategy == 'trigram_heavy':
        # Sử dụng trigrams
        while len(domain) < length:
            if len(domain) < length - 2:
                trigram = random.choice(COMMON_TRIGRAMS)
                domain += trigram
            else:
                domain += random.choice(string.ascii_lowercase)
    
    elif strategy == 'mixed':
        # Mix patterns
        while len(domain) < length:
            choice = random.random()
            if choice < 0.4 and len(domain) < length - 1:
                domain += random.choice(COMMON_BIGRAMS)
            elif choice < 0.7 and len(domain) < length - 2:
                domain += random.choice(COMMON_TRIGRAMS)
            else:
                domain += random.choice(string.ascii_lowercase)
    
    else:  # random
        # Random nhưng có một số pattern
        for i in range(length):
            if i > 0 and random.random() < 0.3:
                # Sử dụng bigram
                if len(domain) < length - 1:
                    domain += random.choice(COMMON_BIGRAMS)
                    break
            domain += random.choice(string.ascii_lowercase)
    
    # Đảm bảo độ dài
    domain = domain[:length]
    while len(domain) < length:
        domain += random.choice(string.ascii_lowercase)
    
    # Thêm số đôi khi
    if random.random() < 0.2:
        pos = random.randint(0, len(domain) - 1)
        domain = domain[:pos] + str(random.randint(0, 9)) + domain[pos+1:]
    
    if add_tld:
        tlds = ['com', 'net', 'org', 'info']
        domain += '.' + random.choice(tlds)
    
    return domain[:length] if not add_tld else domain[:length] + '.' + random.choice(tlds)


def generate_domains(num_domains, start_date=None, add_tld=False):
    """
    Generate neural-like domains
    
    Args:
        num_domains: Số lượng domain
        start_date: Ngày bắt đầu (để tạo seed)
        add_tld: Có thêm TLD không
    """
    if start_date is None:
        start_date = datetime.now()
    
    domains = []
    base_seed = int(start_date.timestamp())
    
    for i in range(num_domains):
        # Seed dựa trên date và index
        seed = base_seed + i
        length = random.randint(10, 18)
        domain = generate_neural_like_domain(length, seed, add_tld)
        domains.append(domain)
    
    return domains

