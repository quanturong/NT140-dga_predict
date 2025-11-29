"""
Time-varying DGA
Thuật toán thay đổi theo thời gian (tuần/tháng)
"""
import random
import string
import hashlib
from datetime import datetime

def wordlist_method(seed):
    """Method 1: Wordlist-based"""
    words = ["secure", "update", "system", "service", "online"]
    word = random.choice(words)
    num = random.randint(100, 999)
    return f"{word}{num}"


def hash_method(seed, date):
    """Method 2: Hash-based"""
    message = f"{date.strftime('%Y%m%d')}-{seed}"
    hash_val = hashlib.md5(message.encode()).hexdigest()[:12]
    return hash_val


def random_method(seed):
    """Method 3: Random"""
    length = random.randint(10, 16)
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def pattern_method(seed, date):
    """Method 4: Pattern-based"""
    date_str = date.strftime('%Y%m%d')
    pattern = f"d{date_str[-4:]}{seed:04d}"
    return pattern


def generate_domains(num_domains, start_date=None, add_tld=False):
    """
    Generate domains với thuật toán thay đổi theo thời gian
    
    Args:
        num_domains: Số lượng domain
        start_date: Ngày bắt đầu
        add_tld: Có thêm TLD không
    """
    if start_date is None:
        start_date = datetime.now()
    
    domains = []
    current_date = start_date
    
    for i in range(num_domains):
        # Xác định method dựa trên tuần
        week_of_year = current_date.isocalendar()[1]
        method_index = week_of_year % 4
        
        # Chọn method
        if method_index == 0:
            domain = wordlist_method(i)
        elif method_index == 1:
            domain = hash_method(i, current_date)
        elif method_index == 2:
            domain = random_method(i)
        else:
            domain = pattern_method(i, current_date)
        
        # Cập nhật date mỗi 100 domain
        if i > 0 and i % 100 == 0:
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

