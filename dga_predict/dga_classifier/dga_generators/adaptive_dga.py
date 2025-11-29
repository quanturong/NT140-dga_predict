"""
Adaptive DGA
Thay đổi strategy dựa trên pattern detection (simulate)
"""
import random
import string
import hashlib
from datetime import datetime

STRATEGIES = ['wordlist', 'hash', 'random', 'pattern', 'mixed']

def wordlist_strategy(seed):
    """Wordlist strategy"""
    words = ["bank", "secure", "update", "system", "service"]
    return f"{random.choice(words)}{random.randint(100, 999)}"


def hash_strategy(seed, date):
    """Hash strategy"""
    message = f"{date.strftime('%Y%m%d')}-{seed}"
    return hashlib.md5(message.encode()).hexdigest()[:14]


def random_strategy(seed):
    """Random strategy"""
    length = random.randint(11, 17)
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def pattern_strategy(seed, date):
    """Pattern strategy"""
    date_part = date.strftime('%m%d')
    return f"d{date_part}{seed:05d}"


def mixed_strategy(seed, date):
    """Mixed strategy"""
    word = random.choice(["secure", "update", "system"])
    hash_part = hashlib.md5(f"{date.strftime('%Y%m%d')}-{seed}".encode()).hexdigest()[:6]
    return f"{word}{hash_part}"


def generate_domains(num_domains, start_date=None, add_tld=False):
    """
    Generate adaptive domains - strategy thay đổi dựa trên "detection"
    
    Args:
        num_domains: Số lượng domain
        start_date: Ngày bắt đầu
        add_tld: Có thêm TLD không
    """
    if start_date is None:
        start_date = datetime.now()
    
    domains = []
    current_date = start_date
    current_strategy = 'hash'  # Bắt đầu với hash
    
    for i in range(num_domains):
        # Thay đổi strategy mỗi 200 domain (simulate adaptation)
        if i > 0 and i % 200 == 0:
            # Rotate strategy
            strategy_index = STRATEGIES.index(current_strategy)
            current_strategy = STRATEGIES[(strategy_index + 1) % len(STRATEGIES)]
        
        # Generate domain theo strategy hiện tại
        if current_strategy == 'wordlist':
            domain = wordlist_strategy(i)
        elif current_strategy == 'hash':
            domain = hash_strategy(i, current_date)
        elif current_strategy == 'random':
            domain = random_strategy(i)
        elif current_strategy == 'pattern':
            domain = pattern_strategy(i, current_date)
        else:  # mixed
            domain = mixed_strategy(i, current_date)
        
        # Cập nhật date
        if i > 0 and i % 110 == 0:
            try:
                current_date = current_date.replace(day=current_date.day + 1)
            except:
                try:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)
                except:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
        
        if add_tld:
            tlds = ['com', 'net', 'org', 'info']
            domain += '.' + random.choice(tlds)
        
        domains.append(domain)
    
    return domains

