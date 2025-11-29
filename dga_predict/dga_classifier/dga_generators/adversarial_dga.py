"""
Adversarial DGA
Tối ưu để bypass ML detection models
Sử dụng techniques để tạo domains trông giống benign nhưng vẫn là DGA
"""
import random
import string
import hashlib
from datetime import datetime

# Benign-like patterns to mimic
BENIGN_LIKE_PATTERNS = {
    'company_suffixes': ['corp', 'inc', 'ltd', 'llc', 'co', 'group', 'tech', 'systems', 'solutions'],
    'common_words': ['global', 'world', 'international', 'digital', 'online', 'network', 'cloud', 'data'],
    'numbers': [100, 200, 300, 500, 1000, 2000, 2024, 2023, 2022],
    'separators': ['', '-', ''],  # Sometimes use hyphen
}

def generate_adversarial_domain(seed, length=12, add_tld=False):
    """
    Generate domain optimized to bypass ML detection
    Combines benign-like patterns with randomness
    """
    random.seed(seed)
    
    # Strategy: make it look like a real company/domain
    strategy = random.choice(['company_style', 'tech_style', 'brand_style', 'mixed_style'])
    
    if strategy == 'company_style':
        # Looks like company domain
        word = random.choice(BENIGN_LIKE_PATTERNS['common_words'])
        suffix = random.choice(BENIGN_LIKE_PATTERNS['company_suffixes'])
        num = random.choice(BENIGN_LIKE_PATTERNS['numbers']) if random.random() < 0.4 else ''
        domain = f"{word}{suffix}{num}".lower()[:length]
    
    elif strategy == 'tech_style':
        # Tech company style
        tech_words = ['cloud', 'data', 'tech', 'digital', 'smart', 'cyber', 'net', 'web']
        word1 = random.choice(tech_words)
        word2 = random.choice(['hub', 'zone', 'lab', 'base', 'core', 'edge'])
        domain = f"{word1}{word2}".lower()[:length]
    
    elif strategy == 'brand_style':
        # Brand-like (short + pronounceable)
        consonants = 'bcdfghjklmnpqrstvwxyz'
        vowels = 'aeiou'
        domain = ""
        for i in range(length):
            if i % 2 == 0:
                domain += random.choice(consonants)
            else:
                domain += random.choice(vowels)
    
    else:  # mixed_style
        # Mix everything
        part1 = random.choice(BENIGN_LIKE_PATTERNS['common_words'])[:4]
        part2 = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 5)))
        num = str(random.randint(10, 99)) if random.random() < 0.3 else ''
        domain = f"{part1}{part2}{num}".lower()[:length]
    
    # Ensure length
    domain = domain[:length]
    while len(domain) < length:
        domain += random.choice(string.ascii_lowercase)
    
    # Add subtle randomness (adversarial noise)
    if random.random() < 0.2:
        pos = random.randint(1, len(domain) - 2)
        domain = domain[:pos] + random.choice(string.ascii_lowercase) + domain[pos+1:]
    
    if add_tld:
        tlds = ['com', 'net', 'org', 'info', 'co']
        domain += '.' + random.choice(tlds)
    
    return domain[:length] if not add_tld else domain[:length] + '.' + random.choice(tlds)


def generate_domains(num_domains, start_date=None, add_tld=False):
    """
    Generate adversarial domains
    
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
        seed = base_seed + i * 7  # More variation
        length = random.randint(9, 15)
        domain = generate_adversarial_domain(seed, length, add_tld)
        domains.append(domain)
    
    return domains

