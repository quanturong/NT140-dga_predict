"""
Hash-based DGA với HMAC/SHA256
Sử dụng cryptographic hash để tạo domain khó dự đoán
"""
import hashlib
import hmac
import random
from datetime import datetime, timedelta

def generate_domain(secret_key, date, domain_index, length=12, add_tld=False):
    """
    Generate domain từ HMAC hash
    
    Args:
        secret_key: Secret key để hash
        date: Date object
        domain_index: Index của domain trong ngày
        length: Độ dài domain
        add_tld: Có thêm TLD không
    """
    # Tạo message từ date và index
    message = f"{date.strftime('%Y-%m-%d')}-{domain_index}"
    
    # HMAC-SHA256
    hmac_hash = hmac.new(
        secret_key.encode() if isinstance(secret_key, str) else secret_key,
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Chuyển đổi hash thành domain
    domain = ""
    valid_chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    
    for i in range(0, min(len(hmac_hash), length * 2), 2):
        byte_val = int(hmac_hash[i:i+2], 16)
        char_index = byte_val % len(valid_chars)
        domain += valid_chars[char_index]
        if len(domain) >= length:
            break
    
    # Đảm bảo độ dài
    while len(domain) < length:
        domain += valid_chars[int(hmac_hash[len(domain) % len(hmac_hash)], 16) % len(valid_chars)]
    
    if add_tld:
        tlds = ['com', 'net', 'org', 'info', 'biz']
        tld_index = int(hmac_hash[-2:], 16) % len(tlds)
        domain += '.' + tlds[tld_index]
    
    return domain[:length] if not add_tld else domain[:length] + '.' + tlds[tld_index]


def generate_domains(num_domains, secret_key="default_secret_key_2024", 
                     start_date=None, add_tld=False):
    """
    Generate multiple hash-based domains
    
    Args:
        num_domains: Số lượng domain
        secret_key: Secret key (có thể thay đổi)
        start_date: Ngày bắt đầu (default: today)
        add_tld: Có thêm TLD không
    """
    if start_date is None:
        start_date = datetime.now()
    
    domains = []
    current_date = start_date
    domain_index = 0
    
    for i in range(num_domains):
        # Mỗi ngày có thể tạo nhiều domain
        if i > 0 and i % 100 == 0:
            current_date += timedelta(days=1)
            domain_index = 0
        
        domain = generate_domain(secret_key, current_date, domain_index, 
                                length=random.randint(10, 20), add_tld=add_tld)
        domains.append(domain)
        domain_index += 1
    
    return domains

