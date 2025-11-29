"""
Context-aware DGA
Sử dụng nhiều nguồn dữ liệu để tạo domain (simulate external data)
"""
import hashlib
import random
from datetime import datetime

# Simulate external data sources
NEWS_KEYWORDS = ["tech", "crypto", "ai", "cloud", "data", "security", "update", "news"]
STOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
WEATHER_TYPES = ["sunny", "cloudy", "rainy", "windy", "stormy"]

def get_context_seed(date):
    """
    Tạo seed từ nhiều nguồn context (simulate)
    Trong thực tế, có thể lấy từ Twitter, news, stock prices, etc.
    """
    # Simulate: dựa trên date để tạo "context"
    day_of_year = date.timetuple().tm_yday
    week_of_year = day_of_year // 7
    
    # Simulate news headline
    news = NEWS_KEYWORDS[day_of_year % len(NEWS_KEYWORDS)]
    
    # Simulate stock price
    stock = STOCK_SYMBOLS[week_of_year % len(STOCK_SYMBOLS)]
    price = 100 + (day_of_year % 200)
    
    # Simulate weather
    weather = WEATHER_TYPES[day_of_year % len(WEATHER_TYPES)]
    temp = 20 + (day_of_year % 30)
    
    # Combine context
    context_str = f"{news}-{stock}-{price}-{weather}-{temp}-{date.strftime('%Y%m%d')}"
    return hashlib.sha256(context_str.encode()).hexdigest()


def generate_domain(context_seed, domain_index, length=14, add_tld=False):
    """
    Generate domain từ context seed
    """
    # Kết hợp context seed với domain index
    combined = f"{context_seed}-{domain_index}"
    hash_val = hashlib.md5(combined.encode()).hexdigest()
    
    # Tạo domain từ hash
    domain = ""
    valid_chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    
    for i in range(0, min(len(hash_val), length * 2), 2):
        byte_val = int(hash_val[i:i+2], 16)
        char_index = byte_val % len(valid_chars)
        domain += valid_chars[char_index]
        if len(domain) >= length:
            break
    
    # Đảm bảo độ dài
    while len(domain) < length:
        domain += valid_chars[int(hash_val[len(domain) % len(hash_val)], 16) % len(valid_chars)]
    
    if add_tld:
        tlds = ['com', 'net', 'org', 'info', 'co']
        tld_index = int(hash_val[-2:], 16) % len(tlds)
        domain += '.' + tlds[tld_index]
    
    return domain[:length] if not add_tld else domain[:length] + '.' + tlds[tld_index]


def generate_domains(num_domains, start_date=None, add_tld=False):
    """
    Generate context-aware domains
    
    Args:
        num_domains: Số lượng domain
        start_date: Ngày bắt đầu
        add_tld: Có thêm TLD không
    """
    if start_date is None:
        start_date = datetime.now()
    
    domains = []
    current_date = start_date
    domain_index = 0
    
    for i in range(num_domains):
        # Mỗi ngày có context khác nhau
        if i > 0 and i % 50 == 0:
            current_date = datetime(current_date.year, current_date.month, current_date.day)
            current_date = current_date.replace(day=current_date.day + 1) if current_date.day < 28 else current_date.replace(month=current_date.month + 1, day=1)
            domain_index = 0
        
        context_seed = get_context_seed(current_date)
        domain = generate_domain(context_seed, domain_index, 
                               length=random.randint(12, 18), add_tld=add_tld)
        domains.append(domain)
        domain_index += 1
    
    return domains

