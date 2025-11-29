"""
Wordlist-based DGA (IcedID/Gozi style)
Kết hợp từ điển với số ngẫu nhiên để tạo domain trông hợp pháp
"""
import random
import string

# Wordlist phổ biến trong DGA
WORDLIST = [
    "bank", "secure", "update", "system", "service", "online", "account",
    "login", "verify", "payment", "support", "admin", "mail", "web",
    "data", "cloud", "tech", "info", "news", "shop", "store", "safe",
    "trust", "guard", "shield", "protect", "access", "portal", "site"
]

NUMBERS = [123, 456, 789, 2024, 2023, 2022, 2021, 100, 200, 300, 500, 1000]

def generate_domains(num_domains, seed=None, add_tld=False):
    """
    Generate wordlist-based domains
    
    Args:
        num_domains: Số lượng domain cần tạo
        seed: Random seed (optional)
        add_tld: Có thêm TLD không
    """
    if seed:
        random.seed(seed)
    
    domains = []
    patterns = [
        lambda: f"{random.choice(WORDLIST)}{random.choice(NUMBERS)}",
        lambda: f"{random.choice(WORDLIST)}{random.choice(WORDLIST)}",
        lambda: f"{random.choice(WORDLIST)}{random.choice(NUMBERS)}{random.choice(WORDLIST)}",
        lambda: f"{random.choice(NUMBERS)}{random.choice(WORDLIST)}",
        lambda: f"{random.choice(WORDLIST)}{random.randint(10, 999)}",
    ]
    
    for _ in range(num_domains):
        pattern = random.choice(patterns)
        domain = pattern()
        
        # Thêm một số ký tự ngẫu nhiên để tăng độ phức tạp
        if random.random() < 0.3:
            domain += random.choice(string.ascii_lowercase)
        
        if add_tld:
            tlds = ['com', 'net', 'org', 'info']
            domain += '.' + random.choice(tlds)
        
        domains.append(domain)
    
    return domains

