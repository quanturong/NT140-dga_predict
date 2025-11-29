"""
Multi-stage/Hybrid DGA
Kết hợp nhiều phương pháp: wordlist + hash + encoding + obfuscation
"""
import hashlib
import base64
import random
import string
from datetime import datetime

WORDLIST = ["secure", "update", "system", "service", "online", "account", "data", "cloud"]

def substitute_chars(text):
    """Character substitution để obfuscate"""
    substitutions = {
        'a': ['a', '4'],
        'e': ['e', '3'],
        'i': ['i', '1'],
        'o': ['o', '0'],
        's': ['s', '5'],
    }
    result = ""
    for char in text:
        if char.lower() in substitutions and random.random() < 0.3:
            result += random.choice(substitutions[char.lower()])
        else:
            result += char
    return result


def generate_domain(date, domain_index, secret="multistage_secret_2024", add_tld=False):
    """
    Multi-stage domain generation:
    1. Wordlist selection
    2. Hash-based transformation
    3. Date-based encoding
    4. Base64 encoding
    5. Character substitution
    """
    # Stage 1: Wordlist
    word = random.choice(WORDLIST)
    
    # Stage 2: Hash-based
    hash_input = f"{word}-{date.strftime('%Y%m%d')}-{domain_index}-{secret}"
    hash_val = hashlib.sha256(hash_input.encode()).hexdigest()
    
    # Stage 3: Date-based transformation
    date_seed = date.strftime("%Y%m%d")
    combined = hash_val[:16] + date_seed
    
    # Stage 4: Base64 encoding (partial)
    encoded = base64.b64encode(combined.encode()).decode()[:12]
    encoded = ''.join(c for c in encoded if c.isalnum()).lower()
    
    # Stage 5: Character substitution
    domain = substitute_chars(encoded)
    
    # Đảm bảo độ dài hợp lý
    if len(domain) < 8:
        domain += hash_val[16:24]
    domain = domain[:20]  # Max length
    
    if add_tld:
        tlds = ['com', 'net', 'org', 'info']
        tld_index = int(hash_val[-2:], 16) % len(tlds)
        domain += '.' + tlds[tld_index]
    
    return domain


def generate_domains(num_domains, start_date=None, secret=None, add_tld=False):
    """
    Generate multi-stage domains
    
    Args:
        num_domains: Số lượng domain
        start_date: Ngày bắt đầu
        secret: Secret key
        add_tld: Có thêm TLD không
    """
    if start_date is None:
        start_date = datetime.now()
    if secret is None:
        secret = "multistage_secret_2024"
    
    domains = []
    current_date = start_date
    domain_index = 0
    
    for i in range(num_domains):
        # Mỗi ngày có thể tạo nhiều domain
        if i > 0 and i % 80 == 0:
            current_date = datetime(current_date.year, current_date.month, current_date.day)
            try:
                current_date = current_date.replace(day=current_date.day + 1)
            except:
                try:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)
                except:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            domain_index = 0
        
        domain = generate_domain(current_date, domain_index, secret, add_tld=add_tld)
        domains.append(domain)
        domain_index += 1
    
    return domains

