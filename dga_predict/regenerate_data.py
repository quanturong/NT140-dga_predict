"""
Script để regenerate data với Real ML-based DGAs được train đúng cách
"""
import os
import sys

# Add path
sys.path.insert(0, os.path.dirname(__file__))

from dga_classifier import data

def is_interactive():
    """Check if running in interactive environment (can use input())"""
    try:
        # Check if stdin is a TTY (terminal)
        return sys.stdin.isatty()
    except:
        # If check fails, assume non-interactive (Kaggle, etc.)
        return False

def main():
    print("=" * 60)
    print("REGENERATING DATA WITH REAL ML-BASED DGAs")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Delete existing traindata.pkl")
    print("  2. Get benign domains FIRST")
    print("  3. Train Real ML-based DGAs on benign domains")
    print("  4. Generate malicious domains using trained models")
    print("  5. Create final dataset")
    print("\n⚠ This may take 10-30 minutes depending on your system...")
    
    # Auto-continue if non-interactive (Kaggle, etc.)
    if not is_interactive():
        print("\n✓ Non-interactive environment detected (Kaggle/Jupyter), auto-continuing...")
    else:
        try:
            response = input("\nContinue? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        except (EOFError, KeyboardInterrupt):
            # If input fails (Kaggle, etc.), auto-continue
            print("\n✓ Input not available, auto-continuing...")
    
    # Delete existing data file
    data_file = "dga_classifier/traindata.pkl"
    if os.path.exists(data_file):
        print(f"\nDeleting {data_file}...")
        os.remove(data_file)
        print("✓ Deleted")
    
    # Also delete model files to force retrain
    import dga_classifier.dga_generators.real_lstm_dga as real_lstm_dga
    import dga_classifier.dga_generators.real_gan_dga as real_gan_dga
    import dga_classifier.dga_generators.adversarial_trained_dga as adversarial_trained_dga
    
    model_files = [
        real_lstm_dga.MODEL_FILE,
        real_lstm_dga.VOCAB_FILE,
        real_gan_dga.MODEL_FILE,
        real_gan_dga.VOCAB_FILE,
        adversarial_trained_dga.MODEL_FILE,
        adversarial_trained_dga.VOCAB_FILE,
    ]
    
    print("\nDeleting existing model files to force retrain...")
    for f in model_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  ✓ Deleted {os.path.basename(f)}")
    
    print("\n" + "=" * 60)
    print("GENERATING DATA...")
    print("=" * 60)
    
    # Generate data with force=True
    data.gen_data(force=True, num_per_dga=15000)
    
    print("\n" + "=" * 60)
    print("✓ DATA GENERATION COMPLETE")
    print("=" * 60)
    print("\nNow you can run: python run.py")
    print("The Real ML-based DGAs should now be harder to detect!")

if __name__ == "__main__":
    main()

