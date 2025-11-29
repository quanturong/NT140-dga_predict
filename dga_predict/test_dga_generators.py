"""
Test script để verify các DGA generators mới hoạt động đúng
"""
import sys
from datetime import datetime

# Test imports
try:
    from dga_classifier.dga_generators import (
        wordlist_dga, hash_dga, context_dga, multistage_dga,
        neural_like_dga, advanced_hash_dga, time_varying_dga,
        obfuscated_dga, adaptive_dga
    )
    print("✓ All DGA generators imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test generation
test_date = datetime(2024, 1, 1)
test_count = 5

print("\nTesting DGA generators...")
print("=" * 60)

try:
    # Test wordlist_dga
    domains = wordlist_dga.generate_domains(test_count)
    print(f"✓ wordlist_dga: {domains[:2]}... ({len(domains)} domains)")
    
    # Test hash_dga
    domains = hash_dga.generate_domains(test_count, start_date=test_date)
    print(f"✓ hash_dga: {domains[:2]}... ({len(domains)} domains)")
    
    # Test context_dga
    domains = context_dga.generate_domains(test_count, start_date=test_date)
    print(f"✓ context_dga: {domains[:2]}... ({len(domains)} domains)")
    
    # Test multistage_dga
    domains = multistage_dga.generate_domains(test_count, start_date=test_date)
    print(f"✓ multistage_dga: {domains[:2]}... ({len(domains)} domains)")
    
    # Test neural_like_dga
    domains = neural_like_dga.generate_domains(test_count, start_date=test_date)
    print(f"✓ neural_like_dga: {domains[:2]}... ({len(domains)} domains)")
    
    # Test advanced_hash_dga
    domains = advanced_hash_dga.generate_domains(test_count, start_date=test_date)
    print(f"✓ advanced_hash_dga: {domains[:2]}... ({len(domains)} domains)")
    
    # Test time_varying_dga
    domains = time_varying_dga.generate_domains(test_count, start_date=test_date)
    print(f"✓ time_varying_dga: {domains[:2]}... ({len(domains)} domains)")
    
    # Test obfuscated_dga
    domains = obfuscated_dga.generate_domains(test_count, start_date=test_date)
    print(f"✓ obfuscated_dga: {domains[:2]}... ({len(domains)} domains)")
    
    # Test adaptive_dga
    domains = adaptive_dga.generate_domains(test_count, start_date=test_date)
    print(f"✓ adaptive_dga: {domains[:2]}... ({len(domains)} domains)")
    
    print("\n" + "=" * 60)
    print("✓ All DGA generators working correctly!")
    
except Exception as e:
    print(f"\n✗ Error testing DGA generators: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test data generation
print("\nTesting data generation...")
try:
    from dga_classifier import data
    
    # Test với số lượng nhỏ
    test_data = data.get_data(force=True, num_per_dga=100)
    print(f"✓ Data generation successful: {len(test_data)} domains")
    
    # Check labels
    labels = [x[0] for x in test_data]
    unique_labels = set(labels)
    print(f"✓ Found {len(unique_labels)} unique DGA families: {sorted(unique_labels)}")
    
    print("\n" + "=" * 60)
    print("✓ System ready for training!")
    
except Exception as e:
    print(f"\n✗ Error in data generation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

