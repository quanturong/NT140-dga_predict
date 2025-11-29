"""
Obfuscated DGA với nhiều lớp encoding
Kết hợp base64, hex, và character substitution
"""
import base64
import hashlib
import random
import string
from datetime import datetime

def obfuscate_domain(domain):
    """Obfuscate domain với character substitution"""
    substitutions = {
        'a': ['a', '4', '@'],
        'e': ['e', '3'],
        'i': ['i', '1', '!'],
        'o': ['o', '0'],
        's': ['s', '5', '$'],
        'l': ['l', '1', '|'],
    }
    result = ""
    for char in domain:
        if char.lower() in substitutions and random.random() < 0.25:
            result += random.choice(substitutions[char.lower()])
        else:
            result += char
    return result


def generate_domains(num_domains, secret_key="obfuscated_key_2024", 
                    start_date=None, add_tld=False):
    """
    Generate obfuscated domains với nhiều lớp encoding
    
    Args:
        num_domains: Số lượng domain
        secret_key: Secret key
        start_date: Ngày bắt đầu
        add_tld: Có thêm TLD không
    """
    if start_date is None:
        start_date = datetime.now()
    
    domains = []
    current_date = start_date
    
    for i in range(num_domains):
        # Tạo input
        date_str = current_date.strftime('%Y%m%d')
        input_str = f"{secret_key}-{date_str}-{i}"
        
        # Layer 1: Hash
        hash_val = hashlib.sha256(input_str.encode()).hexdigest()
        
        # Layer 2: Base64 encoding (partial)
        encoded = base64.b64encode(hash_val[:16].encode()).decode()
        encoded = ''.join(c for c in encoded if c.isalnum()).lower()
        
        # Layer 3: Hex manipulation
        hex_part = hash_val[16:24]
        domain = encoded[:8] + hex_part[:4]
        
        # Layer 4: Obfuscation
        domain = obfuscate_domain(domain)
        
        # Đảm bảo độ dài
        if len(domain) < 10:
            domain += hash_val[24:34]
        domain = domain[:18]
        
        # Cập nhật date
        if i > 0 and i % 90 == 0:
            try:
                current_date = current_date.replace(day=current_date.day + 1)
            except:
                try:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)
                except:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
        
        if add_tld:
            tlds = ['com', 'net', 'org', 'info']
            tld_index = int(hash_val[-2:], 16) % len(tlds)
            domain += '.' + tlds[tld_index]
        
        domains.append(domain)
    
    return domains

