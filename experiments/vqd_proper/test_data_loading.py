"""
Quick test to verify data loading works correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'archive' / 'src'))

from archive.src.loader import load_all_sequences
import numpy as np

print("\n" + "="*60)
print("DATA LOADING TEST")
print("="*60)

# Load sequences
print("\n1. Loading sequences...")
data_path = Path(__file__).parent.parent / "msr_action_data"
sequences, labels = load_all_sequences(str(data_path))

print(f"✅ Loaded {len(sequences)} sequences")
print(f"✅ Labels: {len(labels)} labels")
print(f"✅ Unique classes: {len(set(labels))} (range: {min(labels)}-{max(labels)})")

# Check sequence properties
print("\n2. Sequence statistics:")
lens = [len(seq) for seq in sequences]
print(f"   • Min length: {min(lens)} frames")
print(f"   • Max length: {max(lens)} frames") 
print(f"   • Mean length: {np.mean(lens):.1f} frames")
print(f"   • Feature dim: {sequences[0].shape[1]}D")

# Check class distribution
from collections import Counter
class_counts = Counter(labels)
print(f"\n3. Class distribution:")
print(f"   • Min samples/class: {min(class_counts.values())}")
print(f"   • Max samples/class: {max(class_counts.values())}")
print(f"   • Mean samples/class: {np.mean(list(class_counts.values())):.1f}")

# Test train/test split
from sklearn.model_selection import train_test_split

print("\n4. Testing train/test split (300/60)...")
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels,
    train_size=300,
    test_size=60,
    random_state=42,
    stratify=labels
)

train_counts = Counter(y_train)
test_counts = Counter(y_test)

print(f"   • Train: {len(y_train)} sequences")
print(f"   • Test: {len(y_test)} sequences")
print(f"   • Train classes: {len(set(y_train))}")
print(f"   • Test classes: {len(set(y_test))}")
print(f"   • Train samples/class: {min(train_counts.values())}-{max(train_counts.values())}")
print(f"   • Test samples/class: {min(test_counts.values())}-{max(test_counts.values())}")

# Test frame bank creation
print("\n5. Testing frame bank creation...")
all_frames = np.vstack([seq for seq in X_train])
print(f"   • Frame bank shape: {all_frames.shape}")
print(f"   • Total frames from {len(X_train)} sequences")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - Data loading works correctly!")
print("="*60 + "\n")
