"""
Advanced Hash-based DGA với multiple hash algorithms
Sử dụng SHA256, SHA512, và kết hợp nhiều nguồn input
"""
import hashlib
import random
from datetime import datetime, timedelta

def generate_advanced_hash_domain(secret_key, date, domain_index, 
                                 hash_type='sha256', length=15, add_tld=False):
    """
    Generate domain với advanced hash algorithm
    
    Args:
        secret_key: Secret key
        date: Date object
        domain_index: Index
        hash_type: 'sha256', 'sha512', 'md5'
        length: Domain length
        add_tld: Có thêm TLD không
    """
    # Tạo input phức tạp
    date_str = date.strftime('%Y-%m-%d')
    time_str = date.strftime('%H:%M:%S')
    combined_input = f"{secret_key}-{date_str}-{time_str}-{domain_index}-{hash_type}"
    
    # Hash với algorithm khác nhau
    if hash_type == 'sha256':
        hash_val = hashlib.sha256(combined_input.encode()).hexdigest()
    elif hash_type == 'sha512':
        hash_val = hashlib.sha512(combined_input.encode()).hexdigest()
    else:
        hash_val = hashlib.md5(combined_input.encode()).hexdigest()
    
    # Chuyển đổi hash thành domain với pattern phức tạp
    domain = ""
    valid_chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    
    # Sử dụng nhiều phần của hash
    hash_parts = [hash_val[i:i+4] for i in range(0, len(hash_val), 4)]
    
    for part in hash_parts[:length]:
        byte_val = int(part[:2], 16)
        char_index = byte_val % len(valid_chars)
        domain += valid_chars[char_index]
        if len(domain) >= length:
            break
    
    # Đảm bảo độ dài
    while len(domain) < length:
        remaining = hash_val[len(domain) % len(hash_val):]
        if remaining:
            byte_val = int(remaining[:2], 16) if len(remaining) >= 2 else ord(remaining[0])
            domain += valid_chars[byte_val % len(valid_chars)]
        else:
            domain += random.choice(valid_chars)
    
    if add_tld:
        tlds = ['com', 'net', 'org', 'info', 'biz', 'co']
        tld_index = int(hash_val[-2:], 16) % len(tlds)
        domain += '.' + tlds[tld_index]
    
    return domain[:length] if not add_tld else domain[:length] + '.' + tlds[tld_index]


def generate_domains(num_domains, secret_key=None, start_date=None, 
                     hash_types=None, add_tld=False):
    """
    Generate advanced hash-based domains
    
    Args:
        num_domains: Số lượng domain
        secret_key: Secret key (có thể thay đổi)
        start_date: Ngày bắt đầu
        hash_types: List các hash types để rotate
        add_tld: Có thêm TLD không
    """
    if secret_key is None:
        secret_key = "advanced_secret_key_2024"
    if start_date is None:
        start_date = datetime.now()
    if hash_types is None:
        hash_types = ['sha256', 'sha512', 'md5']
    
    domains = []
    current_date = start_date
    domain_index = 0
    
    for i in range(num_domains):
        # Rotate hash types
        hash_type = hash_types[i % len(hash_types)]
        
        # Mỗi ngày có thể tạo nhiều domain
        if i > 0 and i % 120 == 0:
            current_date += timedelta(days=1)
            domain_index = 0
        
        length = random.randint(12, 20)
        domain = generate_advanced_hash_domain(
            secret_key, current_date, domain_index, 
            hash_type, length, add_tld
        )
        domains.append(domain)
        domain_index += 1
    
    return domains

