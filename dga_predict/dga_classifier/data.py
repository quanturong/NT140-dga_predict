"""Generates data for train/test algorithms (~1M domains, Endgame-style)"""
from datetime import datetime
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import pickle, os, random, string, tldextract

from dga_classifier.dga_generators import (
    # Strong DGAs
    hash_dga, context_dga, multistage_dga,
    neural_like_dga, advanced_hash_dga, time_varying_dga,
    obfuscated_dga, adaptive_dga,
    # REAL ML-based DGAs (state-of-the-art)
    real_lstm_dga, real_gan_dga, adversarial_trained_dga
)

ALEXA_1M = "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"
DATA_FILE = "traindata.pkl"


def get_alexa(num, address=ALEXA_1M, filename="top-1m.csv"):
    """Download Alexa top domains or generate synthetic benign ones if offline"""
    print(f"Downloading Alexa top {num} domains...")
    try:
        url = urlopen(address, timeout=10)
        zipfile = ZipFile(BytesIO(url.read()))
        data = zipfile.read(filename).decode("utf-8")

        domains = []
        for line in data.strip().split("\n"):
            if len(domains) >= num:
                break
            try:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    domain = tldextract.extract(parts[1].strip()).domain
                    if domain:
                        domains.append(domain)
            except Exception:
                continue

        if len(domains) > 0:
            print(f"✓ Downloaded {len(domains)} benign domains from Alexa")
            return domains
        raise Exception("No domains extracted")

    except Exception as e:
        # 🔧 Improved synthetic benign domain generator — more realistic
        print(f"⚠ Alexa download failed: {e}")
        print("Using improved synthetic benign domain generator...")

        # Common prefixes (more diverse)
        prefixes = [
            "tech","smart","daily","news","blog","cloud","auto","shop","food",
            "data","crypto","home","media","travel","game","life","pro","global",
            "local","green","eco","trend","world","info","secure","digital",
            "online","mobile","social","business","market","finance","health",
            "education","science","music","video","photo","design","creative"
        ]
        
        # Common suffixes
        suffixes = ["hub","zone","site","lab","net","store","world","plus","go","page",
                   "app","link","space","base","core","edge","flow","grid","node"]
        
        # Common words (to differentiate from wordlist DGA)
        common_words = [
            "company","service","solutions","systems","network","platform",
            "group","team","studio","agency","center","center","office",
            "enterprise","corp","inc","ltd","co","org","com"
        ]
        
        # Common patterns in real domains
        patterns = [
            # Pattern 1: prefix + suffix (most common)
            lambda: random.choice(prefixes) + random.choice(suffixes),
            # Pattern 2: word + number (but different from DGA)
            lambda: random.choice(common_words) + str(random.randint(1000, 9999)),
            # Pattern 3: two words
            lambda: random.choice(prefixes) + random.choice(common_words[:10]),
            # Pattern 4: word + suffix
            lambda: random.choice(common_words[:15]) + random.choice(suffixes),
            # Pattern 5: prefix + number (4 digits, different from DGA)
            lambda: random.choice(prefixes) + str(random.randint(1000, 9999)),
            # Pattern 6: pronounceable random (using common bigrams)
            lambda: generate_pronounceable_domain(),
            # Pattern 7: brand-like (short + suffix)
            lambda: ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 5))) + random.choice(suffixes),
        ]

        result = []
        for _ in range(num):
            pattern = random.choice(patterns)
            name = pattern()
            # Ensure reasonable length
            if len(name) < 4:
                name += random.choice(suffixes)
            if len(name) > 25:
                name = name[:25]
            result.append(name)

        print(f"✓ Generated {len(result)} synthetic benign domains")
        return result


def generate_pronounceable_domain():
    """Generate pronounceable domain using common letter combinations"""
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    
    # Common consonant-vowel patterns
    patterns = [
        lambda: random.choice(consonants) + random.choice(vowels) + random.choice(consonants) + random.choice(vowels),
        lambda: random.choice(consonants) + random.choice(vowels) + random.choice(consonants),
        lambda: random.choice(vowels) + random.choice(consonants) + random.choice(vowels),
    ]
    
    domain = ""
    length = random.randint(6, 12)
    
    for i in range(length):
        if i % 2 == 0:
            domain += random.choice(consonants)
        else:
            domain += random.choice(vowels)
    
    # Sometimes add a consonant cluster
    if random.random() < 0.3 and len(domain) < length:
        domain += random.choice(['st', 'nd', 'rd', 'th', 'ch', 'sh'])
    
    return domain[:length]


def gen_malicious_core(num_per_dga=15000):
    """Generate 11 STRONG DGA families (removed medium ones, added strongest)"""
    domains, labels = [], []
    print(f"Generating STRONG DGAs ({num_per_dga} per family)...")

    # 1. Hash-based DGA (HMAC/SHA256) - Strong
    print("  - Generating hash-based DGA...")
    secret_keys = ["secret_key_1", "secret_key_2", "secret_key_3"]
    segs = max(1, num_per_dga // len(secret_keys))
    for key in secret_keys:
        domains += hash_dga.generate_domains(segs, secret_key=key, start_date=datetime(2024,1,1))
        labels += ["hash_dga"] * segs

    # 2. Enhanced Context-aware DGA - Very Strong
    print("  - Generating enhanced context-aware DGA...")
    domains += context_dga.generate_domains(num_per_dga, start_date=datetime(2024,1,1))
    labels += ["context_dga"] * num_per_dga

    # 3. Multi-stage DGA - Very Strong
    print("  - Generating multi-stage DGA...")
    secrets = ["multistage_secret_1", "multistage_secret_2"]
    segs = max(1, num_per_dga // len(secrets))
    for secret in secrets:
        domains += multistage_dga.generate_domains(segs, secret=secret, start_date=datetime(2024,1,1))
        labels += ["multistage_dga"] * segs

    # 4. Neural-like DGA - Strong
    print("  - Generating neural-like DGA...")
    domains += neural_like_dga.generate_domains(num_per_dga, start_date=datetime(2024,1,1))
    labels += ["neural_like_dga"] * num_per_dga

    # 5. Advanced Hash DGA - Very Strong
    print("  - Generating advanced hash DGA...")
    domains += advanced_hash_dga.generate_domains(num_per_dga, start_date=datetime(2024,1,1))
    labels += ["advanced_hash_dga"] * num_per_dga

    # 6. Time-varying DGA - Strong
    print("  - Generating time-varying DGA...")
    domains += time_varying_dga.generate_domains(num_per_dga, start_date=datetime(2024,1,1))
    labels += ["time_varying_dga"] * num_per_dga

    # 7. Obfuscated DGA - Very Strong
    print("  - Generating obfuscated DGA...")
    domains += obfuscated_dga.generate_domains(num_per_dga, start_date=datetime(2024,1,1))
    labels += ["obfuscated_dga"] * num_per_dga

    # 8. Adaptive DGA - Very Strong
    print("  - Generating adaptive DGA...")
    domains += adaptive_dga.generate_domains(num_per_dga, start_date=datetime(2024,1,1))
    labels += ["adaptive_dga"] * num_per_dga

    # 9. Real LSTM-based DGA - STATE-OF-THE-ART (NEW)
    print("  - Generating real LSTM-based DGA (training on benign domains)...")
    domains += real_lstm_dga.generate_domains(num_per_dga, start_date=datetime(2024,1,1))
    labels += ["real_lstm_dga"] * num_per_dga

    # 10. Real GAN-based DGA - STATE-OF-THE-ART (NEW)
    print("  - Generating real GAN-based DGA (training GAN)...")
    domains += real_gan_dga.generate_domains(num_per_dga, start_date=datetime(2024,1,1))
    labels += ["real_gan_dga"] * num_per_dga

    # 11. Adversarial-trained DGA - STATE-OF-THE-ART (NEW)
    print("  - Generating adversarial-trained DGA (bypassing detector)...")
    domains += adversarial_trained_dga.generate_domains(num_per_dga, start_date=datetime(2024,1,1))
    labels += ["adversarial_trained_dga"] * num_per_dga

    print(f"✓ Generated {len(domains)} malicious domains (STRONG DGAs only)")
    return domains, labels


def gen_additional_dgas(num_per_dga=15000):
    """Generate 19 stub DGA families"""
    families = [
        'bamital','bedep','beebone','chinad','cryptowall','dyre',
        'emotet','feodo','fobber','gameover','gozi','hesperbot',
        'matsnu','mirai','necurs','nymaim','proslikefan','pushdo','suppobox'
    ]
    print(f"Generating {len(families)} stub DGAs ({num_per_dga} each)...")
    domains, labels = [], []

    def randdom(n=12): return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))

    for fam in families:
        for _ in range(num_per_dga):
            if fam == "suppobox":
                d = randdom(5)+randdom(5)
            elif fam in ("mirai","gozi","emotet"):
                d = randdom(8)+str(random.randint(10,99))
            else:
                d = randdom(12)
            domains.append(d)
            labels.append(fam)
    print(f"✓ Generated {len(domains)} stub malicious domains")
    return domains, labels


def gen_data(force=False, num_per_dga=15000):
    """Generate ~1M domain dataset (30 DGAs * 15k + same benign)"""
    if force or not os.path.isfile(DATA_FILE):
        print("\n" + "="*60)
        print("GENERATING TRAINING DATA (~1M domains)")
        print("="*60)

        d1, l1 = gen_malicious_core(num_per_dga)
        d2, l2 = gen_additional_dgas(num_per_dga)
        domains, labels = d1 + d2, l1 + l2

        benign = get_alexa(len(domains))
        domains += benign
        labels += ["benign"] * len(benign)

        # normalize
        domains = [tldextract.extract(d).domain.lower() for d in domains]
        data = list(zip(labels, domains))
        random.shuffle(data)

        print(f"\n✓ Total domains: {len(domains)}")
        print(f"  - Malicious: {len([x for x in labels if x!='benign'])}")
        print(f"  - Benign: {labels.count('benign')}")

        print(f"\nSaving to {DATA_FILE}...")
        with open(DATA_FILE, "wb") as f:
            pickle.dump(data, f)
        print("✓ Data saved successfully\n")
    else:
        print(f"{DATA_FILE} already exists — use get_data(force=True) to regenerate.")


def get_data(force=False, num_per_dga=15000):
    """Return dataset as list of (label, domain)"""
    gen_data(force=force, num_per_dga=num_per_dga)
    with open(DATA_FILE, "rb") as f:
        return pickle.load(f)
