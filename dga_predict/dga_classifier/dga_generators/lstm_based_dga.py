"""
LSTM-based DGA Generator
Sử dụng LSTM để generate domains giống như benign domains
Trained trên patterns từ benign domains
"""
import random
import string
import numpy as np
from datetime import datetime

# Character vocabulary
CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789'
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

# Common patterns learned from benign domains (simulated LSTM knowledge)
BENIGN_PATTERNS = {
    'start_chars': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
    'common_bigrams': ['th', 'he', 'in', 'er', 'an', 're', 'ed', 'on', 'es', 'st', 'en', 'at', 'to', 'nt', 'nd'],
    'common_trigrams': ['the', 'and', 'ing', 'ion', 'tio', 'ent', 'ati', 'for', 'her', 'ter'],
    'vowel_consonant_patterns': ['cvc', 'cvcc', 'ccvc', 'cvcv', 'vcvc'],
    'endings': ['er', 'ly', 'ed', 'ing', 'ion', 'tion', 'sion', 'ment', 'ness', 'able', 'ible']
}

def generate_lstm_like_sequence(length, seed=None):
    """
    Generate sequence using LSTM-like patterns
    Simulates what an LSTM would generate after training on benign domains
    """
    if seed:
        random.seed(seed)
    
    sequence = ""
    
    # Strategy: mix different patterns like LSTM would
    strategy = random.choice(['pattern_heavy', 'bigram_heavy', 'pronounceable', 'mixed'])
    
    if strategy == 'pattern_heavy':
        # Use common patterns
        while len(sequence) < length:
            if len(sequence) == 0:
                sequence += random.choice(BENIGN_PATTERNS['start_chars'])
            elif len(sequence) < length - 2:
                if random.random() < 0.4:
                    sequence += random.choice(BENIGN_PATTERNS['common_bigrams'])
                elif random.random() < 0.6:
                    sequence += random.choice(BENIGN_PATTERNS['common_trigrams'])
                else:
                    sequence += random.choice(string.ascii_lowercase)
            else:
                sequence += random.choice(string.ascii_lowercase)
    
    elif strategy == 'bigram_heavy':
        # Heavy use of common bigrams
        while len(sequence) < length:
            if len(sequence) < length - 1:
                bigram = random.choice(BENIGN_PATTERNS['common_bigrams'])
                sequence += bigram
            else:
                sequence += random.choice(string.ascii_lowercase)
    
    elif strategy == 'pronounceable':
        # Pronounceable pattern (CVC, CVCC, etc.)
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvwxyz'
        pattern = random.choice(BENIGN_PATTERNS['vowel_consonant_patterns'])
        
        for char_type in pattern:
            if char_type == 'c':
                sequence += random.choice(consonants)
            elif char_type == 'v':
                sequence += random.choice(vowels)
            
            if len(sequence) >= length:
                break
        
        # Fill remaining
        while len(sequence) < length:
            if len(sequence) % 2 == 0:
                sequence += random.choice(consonants)
            else:
                sequence += random.choice(vowels)
    
    else:  # mixed
        # Mix all strategies
        while len(sequence) < length:
            choice = random.random()
            if choice < 0.3 and len(sequence) < length - 1:
                sequence += random.choice(BENIGN_PATTERNS['common_bigrams'])
            elif choice < 0.5 and len(sequence) < length - 2:
                sequence += random.choice(BENIGN_PATTERNS['common_trigrams'])
            elif choice < 0.7 and len(sequence) > 0:
                # Add ending pattern
                if len(sequence) < length - 2:
                    ending = random.choice(BENIGN_PATTERNS['endings'])
                    sequence += ending
            else:
                sequence += random.choice(string.ascii_lowercase)
    
    # Ensure length
    sequence = sequence[:length]
    while len(sequence) < length:
        sequence += random.choice(string.ascii_lowercase)
    
    # Sometimes add numbers (like real domains)
    if random.random() < 0.15:
        pos = random.randint(0, len(sequence) - 1)
        sequence = sequence[:pos] + str(random.randint(0, 9)) + sequence[pos+1:]
    
    return sequence[:length]


def generate_domains(num_domains, start_date=None, add_tld=False):
    """
    Generate LSTM-based domains
    
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
        seed = base_seed + i
        length = random.randint(8, 16)
        domain = generate_lstm_like_sequence(length, seed)
        
        if add_tld:
            tlds = ['com', 'net', 'org', 'info']
            domain += '.' + random.choice(tlds)
        
        domains.append(domain)
    
    return domains

