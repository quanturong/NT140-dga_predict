"""
Script để verify xem Real ML-based DGAs có thực sự train và generate đúng không
"""
import os
import sys
from datetime import datetime

# Add path
sys.path.insert(0, os.path.dirname(__file__))

from dga_classifier.dga_generators import real_lstm_dga, real_gan_dga, adversarial_trained_dga

def check_model_files():
    """Kiểm tra xem model files có tồn tại không"""
    print("=" * 60)
    print("CHECKING MODEL FILES")
    print("=" * 60)
    
    lstm_model = real_lstm_dga.MODEL_FILE
    lstm_vocab = real_lstm_dga.VOCAB_FILE
    gan_model = real_gan_dga.MODEL_FILE
    gan_vocab = real_gan_dga.VOCAB_FILE
    adv_model = adversarial_trained_dga.MODEL_FILE
    adv_vocab = adversarial_trained_dga.VOCAB_FILE
    
    files = [
        ("LSTM Model", lstm_model),
        ("LSTM Vocab", lstm_vocab),
        ("GAN Model", gan_model),
        ("GAN Vocab", gan_vocab),
        ("Adversarial Model", adv_model),
        ("Adversarial Vocab", adv_vocab),
    ]
    
    for name, path in files:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"{name:20s}: {'✓ EXISTS' if exists else '✗ NOT FOUND'} ({size:,} bytes)")
    
    return any(os.path.exists(f[1]) for f in files)


def test_generation():
    """Test generate một số domains từ mỗi Real ML-based DGA"""
    print("\n" + "=" * 60)
    print("TESTING DOMAIN GENERATION")
    print("=" * 60)
    
    test_count = 10
    test_date = datetime(2024, 1, 1)
    
    # Test LSTM
    print("\n1. Real LSTM DGA:")
    try:
        domains = real_lstm_dga.generate_domains(test_count, start_date=test_date)
        print(f"   Generated {len(domains)} domains")
        print(f"   Samples: {domains[:5]}")
        
        # Check if they look random (fallback) or realistic
        avg_length = sum(len(d) for d in domains) / len(domains)
        print(f"   Avg length: {avg_length:.1f}")
        
        # Check entropy (random domains have high entropy)
        from dga_classifier.manual_rf import shannon_entropy
        avg_entropy = sum(shannon_entropy(d) for d in domains) / len(domains)
        print(f"   Avg entropy: {avg_entropy:.2f} (higher = more random)")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test GAN
    print("\n2. Real GAN DGA:")
    try:
        domains = real_gan_dga.generate_domains(test_count, start_date=test_date)
        print(f"   Generated {len(domains)} domains")
        print(f"   Samples: {domains[:5]}")
        
        avg_length = sum(len(d) for d in domains) / len(domains)
        print(f"   Avg length: {avg_length:.1f}")
        
        from dga_classifier.manual_rf import shannon_entropy
        avg_entropy = sum(shannon_entropy(d) for d in domains) / len(domains)
        print(f"   Avg entropy: {avg_entropy:.2f}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test Adversarial
    print("\n3. Adversarial-trained DGA:")
    try:
        domains = adversarial_trained_dga.generate_domains(test_count, start_date=test_date)
        print(f"   Generated {len(domains)} domains")
        print(f"   Samples: {domains[:5]}")
        
        avg_length = sum(len(d) for d in domains) / len(domains)
        print(f"   Avg length: {avg_length:.1f}")
        
        from dga_classifier.manual_rf import shannon_entropy
        avg_entropy = sum(shannon_entropy(d) for d in domains) / len(domains)
        print(f"   Avg entropy: {avg_entropy:.2f}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")


def check_data_file():
    """Kiểm tra traindata.pkl"""
    print("\n" + "=" * 60)
    print("CHECKING DATA FILE")
    print("=" * 60)
    
    data_file = "dga_predict/dga_classifier/traindata.pkl"
    if os.path.exists(data_file):
        size = os.path.getsize(data_file)
        print(f"traindata.pkl: ✓ EXISTS ({size:,} bytes)")
        print(f"\n⚠ NOTE: If you want to regenerate with new code, delete this file")
        print(f"   or run: python -c 'from dga_classifier import data; data.gen_data(force=True)'")
    else:
        print(f"traindata.pkl: ✗ NOT FOUND")
        print(f"   Data will be generated on first run")


def main():
    print("\n" + "=" * 60)
    print("REAL ML-BASED DGAs VERIFICATION")
    print("=" * 60)
    
    check_data_file()
    has_models = check_model_files()
    test_generation()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if not has_models:
        print("⚠ No model files found. Real ML-based DGAs will train on first use.")
        print("  This may take several minutes.")
    else:
        print("✓ Model files exist. Real ML-based DGAs should use trained models.")
    
    print("\n💡 To regenerate data with Real ML-based DGAs:")
    print("   1. Delete traindata.pkl")
    print("   2. Run: python run.py --force")
    print("   3. Look for messages like 'Training LSTM generator...' in output")


if __name__ == "__main__":
    main()

