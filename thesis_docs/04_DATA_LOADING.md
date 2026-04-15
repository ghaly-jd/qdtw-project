# 04 - Data Loading

**File:** `04_DATA_LOADING.md`  
**Purpose:** Detailed implementation of MSR Action3D data loading  
**For Thesis:** Methodology chapter - data preprocessing

---

## 4.1 Overview

Data loading is the first stage of our pipeline, responsible for:
1. Parsing skeleton text files
2. Extracting 3D joint positions
3. Organizing sequences with metadata
4. Quality validation

**Input:** Raw `*_skeleton.txt` files  
**Output:** List of sequences (each: ndarray of shape `(T, 60)`)

---

## 4.2 File Format Parsing

### 4.2.1 Skeleton File Structure

Each file contains:
```
Line 1: <frame_count>          # Number of frames T
Line 2: <joint_count>           # Always 20
Lines 3+: Joint data (20 × T lines)
```

**Per joint (12 values):**
```
<x> <y> <z> <depthX> <depthY> <colorX> <colorY> <quatW> <quatX> <quatY> <quatZ> <trackingState>
```

**We extract only:** `x`, `y`, `z`, `trackingState`

### 4.2.2 Implementation

```python
import numpy as np

def load_skeleton_file(filepath):
    """
    Parse MSR Action3D skeleton file.
    
    Parameters
    ----------
    filepath : str
        Path to *_skeleton.txt file
    
    Returns
    -------
    skeleton_data : ndarray, shape (n_frames, 20, 4)
        Joint data: (frame, joint, [x, y, z, confidence])
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    n_frames = int(lines[0].strip())
    n_joints = int(lines[1].strip())
    
    if n_joints != 20:
        raise ValueError(f"Expected 20 joints, got {n_joints}")
    
    # Initialize storage
    skeleton_data = np.zeros((n_frames, n_joints, 4))
    
    # Parse joint data
    line_idx = 2
    for frame_idx in range(n_frames):
        for joint_idx in range(n_joints):
            if line_idx >= len(lines):
                raise ValueError(f"Unexpected EOF at frame {frame_idx}, joint {joint_idx}")
            
            values = lines[line_idx].strip().split()
            
            # Extract x, y, z (first 3 values)
            x = float(values[0])
            y = float(values[1])
            z = float(values[2])
            
            # Extract tracking state (last value)
            tracking_state = int(values[-1])
            
            # Convert tracking state to confidence
            # 0 = not tracked, 1 = inferred, 2 = tracked
            confidence = 1.0 if tracking_state == 2 else 0.5 if tracking_state == 1 else 0.0
            
            skeleton_data[frame_idx, joint_idx] = [x, y, z, confidence]
            line_idx += 1
    
    return skeleton_data
```

---

## 4.3 Sequence Flattening

### 4.3.1 From 3D Skeleton to Feature Vector

Transform `(T, 20, 3)` → `(T, 60)`:

```python
def flatten_sequence(skeleton_data):
    """
    Flatten 3D skeleton to feature vectors.
    
    Parameters
    ----------
    skeleton_data : ndarray, shape (T, 20, 4)
        Skeleton with confidence scores
    
    Returns
    -------
    flat_sequence : ndarray, shape (T, 60)
        Flattened positions (20 joints × 3 coords)
    """
    # Extract only x, y, z (drop confidence)
    positions = skeleton_data[:, :, :3]  # (T, 20, 3)
    
    # Flatten: (T, 20, 3) → (T, 60)
    flat_sequence = positions.reshape(len(positions), -1)
    
    return flat_sequence
```

**Resulting feature order:**
```
Features 0-2:   Joint 0 (HipCenter):       x₀, y₀, z₀
Features 3-5:   Joint 1 (Spine):           x₁, y₁, z₁
Features 6-8:   Joint 2 (ShoulderCenter):  x₂, y₂, z₂
...
Features 57-59: Joint 19 (FootRight):      x₁₉, y₁₉, z₁₉
```

---

## 4.4 Dataset Loading

### 4.4.1 Filename Parsing

Extract metadata from filename: `a{action}_s{subject}_e{execution}_skeleton.txt`

```python
import os
import re

def parse_filename(filename):
    """
    Extract metadata from MSR Action3D filename.
    
    Parameters
    ----------
    filename : str
        Example: 'a01_s05_e02_skeleton.txt'
    
    Returns
    -------
    metadata : dict
        {'action': 1, 'subject': 5, 'execution': 2}
        Or None if pattern doesn't match
    """
    pattern = r'a(\d{2})_s(\d{2})_e(\d{2})_skeleton\.txt'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    return {
        'action': int(match.group(1)),
        'subject': int(match.group(2)),
        'execution': int(match.group(3))
    }
```

### 4.4.2 Full Dataset Loader

```python
def load_all_sequences(data_dir, verbose=True):
    """
    Load entire MSR Action3D dataset.
    
    Parameters
    ----------
    data_dir : str
        Directory containing *_skeleton.txt files
    verbose : bool
        Print progress
    
    Returns
    -------
    sequences : list of ndarray
        Each element: shape (T_i, 60)
    labels : list of int
        Action class labels (0-19)
    metadata : list of dict
        File metadata (action, subject, execution)
    """
    sequences = []
    labels = []
    metadata = []
    
    # List all skeleton files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('_skeleton.txt')]
    
    if verbose:
        print(f"Found {len(all_files)} skeleton files in {data_dir}")
    
    # Load each file
    for i, filename in enumerate(sorted(all_files)):
        # Parse metadata
        meta = parse_filename(filename)
        if meta is None:
            if verbose:
                print(f"Skipping {filename} (pattern mismatch)")
            continue
        
        # Load skeleton
        filepath = os.path.join(data_dir, filename)
        try:
            skeleton_data = load_skeleton_file(filepath)
        except Exception as e:
            if verbose:
                print(f"Error loading {filename}: {e}")
            continue
        
        # Flatten to feature vectors
        flat_sequence = flatten_sequence(skeleton_data)
        
        # Convert action ID to 0-indexed label
        label = meta['action'] - 1  # a01 → 0, a02 → 1, ..., a20 → 19
        
        # Store
        sequences.append(flat_sequence)
        labels.append(label)
        metadata.append({
            'filename': filename,
            'action': meta['action'],
            'subject': meta['subject'],
            'execution': meta['execution'],
            'n_frames': len(flat_sequence)
        })
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(all_files)} files...")
    
    if verbose:
        print(f"\nSuccessfully loaded {len(sequences)} sequences")
        print(f"  Classes: {len(set(labels))} unique actions")
        print(f"  Subjects: {len(set(m['subject'] for m in metadata))} subjects")
        print(f"  Total frames: {sum(len(s) for s in sequences)}")
    
    return sequences, labels, metadata
```

---

## 4.5 Quality Validation

### 4.5.1 Checking Tracking Quality

Filter sequences with poor tracking:

```python
def validate_sequence_quality(skeleton_data, min_confidence=0.9):
    """
    Check if sequence has sufficient tracking quality.
    
    Parameters
    ----------
    skeleton_data : ndarray, shape (T, 20, 4)
        Skeleton with confidence scores
    min_confidence : float
        Minimum fraction of well-tracked frames
    
    Returns
    -------
    is_valid : bool
        True if sequence passes quality check
    stats : dict
        Quality statistics
    """
    # Extract confidence scores (4th column)
    confidence = skeleton_data[:, :, 3]  # (T, 20)
    
    # Frame is "well-tracked" if >90% of joints tracked
    well_tracked_frames = (confidence >= 1.0).sum(axis=1) >= 18  # 18/20 joints
    
    fraction_good = well_tracked_frames.sum() / len(well_tracked_frames)
    
    stats = {
        'total_frames': len(skeleton_data),
        'well_tracked_frames': well_tracked_frames.sum(),
        'fraction_good': fraction_good,
        'is_valid': fraction_good >= min_confidence
    }
    
    return stats['is_valid'], stats
```

### 4.5.2 Filtering Dataset

```python
def load_and_filter_sequences(data_dir, min_confidence=0.9, verbose=True):
    """
    Load sequences with quality filtering.
    
    Returns only sequences that pass quality validation.
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith('_skeleton.txt')]
    
    sequences = []
    labels = []
    metadata = []
    rejected = []
    
    for filename in sorted(all_files):
        # Parse metadata
        meta = parse_filename(filename)
        if meta is None:
            continue
        
        # Load skeleton (with confidence)
        filepath = os.path.join(data_dir, filename)
        skeleton_data = load_skeleton_file(filepath)
        
        # Validate quality
        is_valid, stats = validate_sequence_quality(skeleton_data, min_confidence)
        
        if is_valid:
            # Good quality - keep it
            flat_sequence = flatten_sequence(skeleton_data)
            sequences.append(flat_sequence)
            labels.append(meta['action'] - 1)
            metadata.append(meta)
        else:
            # Poor quality - reject
            rejected.append({
                'filename': filename,
                'fraction_good': stats['fraction_good']
            })
    
    if verbose:
        print(f"Loaded {len(sequences)} sequences (rejected {len(rejected)})")
        if rejected:
            print(f"  Worst quality: {min(r['fraction_good'] for r in rejected):.1%}")
    
    return sequences, labels, metadata, rejected
```

**In our experiments:** All 567 sequences pass quality check (min_confidence=0.9)

---

## 4.6 Train-Test Splitting

### 4.6.1 Cross-Subject Split

```python
def split_by_subject(sequences, labels, metadata, test_subject):
    """
    Split dataset by subject (cross-subject evaluation).
    
    Parameters
    ----------
    sequences, labels, metadata : dataset
    test_subject : int
        Subject ID to hold out (1-10)
    
    Returns
    -------
    train_sequences, train_labels : training set
    test_sequences, test_labels : test set
    """
    train_sequences, train_labels = [], []
    test_sequences, test_labels = [], []
    
    for seq, label, meta in zip(sequences, labels, metadata):
        if meta['subject'] == test_subject:
            test_sequences.append(seq)
            test_labels.append(label)
        else:
            train_sequences.append(seq)
            train_labels.append(label)
    
    print(f"Split by subject {test_subject}:")
    print(f"  Train: {len(train_sequences)} sequences")
    print(f"  Test:  {len(test_sequences)} sequences")
    
    return train_sequences, train_labels, test_sequences, test_labels
```

**Our configuration:** Subject 5 as fixed test set (~10% holdout)

---

## 4.7 Usage Example

### 4.7.1 Complete Loading Pipeline

```python
# ═══════════════════════════════════════════════════════════
# EXAMPLE: Load MSR Action3D dataset
# ═══════════════════════════════════════════════════════════

from archive.src.loader import load_all_sequences, split_by_subject

# 1. Load all sequences
data_dir = '/path/to/qdtw_project/msr_action_data/'
sequences, labels, metadata = load_all_sequences(data_dir, verbose=True)

# Output:
# Found 567 skeleton files in ...
# Loaded 50/567 files...
# Loaded 100/567 files...
# ...
# Successfully loaded 567 sequences
#   Classes: 20 unique actions
#   Subjects: 10 subjects
#   Total frames: 24,010

# 2. Inspect first sequence
print(f"\nFirst sequence: {metadata[0]['filename']}")
print(f"  Shape: {sequences[0].shape}")  # (T, 60)
print(f"  Action: {labels[0]} (class {labels[0]})")
print(f"  Subject: {metadata[0]['subject']}")
print(f"  Duration: {len(sequences[0])} frames")

# 3. Split by subject
train_seqs, train_labels, test_seqs, test_labels = split_by_subject(
    sequences, labels, metadata, test_subject=5
)

# Output:
# Split by subject 5:
#   Train: 510 sequences
#   Test:  57 sequences

# 4. Check class distribution
from collections import Counter
print("\nTrain class distribution:")
train_counts = Counter(train_labels)
for class_id in sorted(train_counts.keys()):
    print(f"  Class {class_id:2d}: {train_counts[class_id]:3d} sequences")
```

---

## 4.8 Data Statistics

### 4.8.1 Sequence Length Distribution

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute statistics
lengths = [len(seq) for seq in sequences]

print("Sequence length statistics:")
print(f"  Min:    {np.min(lengths)} frames")
print(f"  Max:    {np.max(lengths)} frames")
print(f"  Mean:   {np.mean(lengths):.1f} frames")
print(f"  Median: {np.median(lengths):.0f} frames")
print(f"  Std:    {np.std(lengths):.1f} frames")

# Histogram
plt.figure(figsize=(10, 4))
plt.hist(lengths, bins=30, edgecolor='black')
plt.xlabel('Sequence Length (frames)')
plt.ylabel('Count')
plt.title('MSR Action3D Sequence Length Distribution')
plt.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
plt.legend()
plt.tight_layout()
plt.savefig('figures/sequence_lengths.png', dpi=300)
```

**Output:**
```
Sequence length statistics:
  Min:    13 frames
  Max:    255 frames
  Mean:   42.3 frames
  Median: 38 frames
  Std:    18.7 frames
```

### 4.8.2 Per-Class Statistics

```python
# Analyze per-class sequence lengths
import pandas as pd

class_stats = []
for class_id in range(20):
    class_seqs = [sequences[i] for i, l in enumerate(labels) if l == class_id]
    class_lengths = [len(s) for s in class_seqs]
    
    class_stats.append({
        'class': class_id,
        'count': len(class_seqs),
        'mean_length': np.mean(class_lengths),
        'std_length': np.std(class_lengths)
    })

df = pd.DataFrame(class_stats)
print(df.to_string(index=False))
```

---

## 4.9 Error Handling

### 4.9.1 Common Issues

**1. Corrupted Files:**
```python
try:
    skeleton_data = load_skeleton_file(filepath)
except Exception as e:
    print(f"ERROR: {filepath}")
    print(f"  {type(e).__name__}: {e}")
    # Skip this file or use fallback
```

**2. Incomplete Sequences:**
```python
# Check for premature EOF
expected_lines = 2 + (n_frames * n_joints)
actual_lines = len(lines)
if actual_lines < expected_lines:
    raise ValueError(f"Expected {expected_lines} lines, got {actual_lines}")
```

**3. Invalid Values:**
```python
# Check for NaN or inf
if np.any(~np.isfinite(flat_sequence)):
    print(f"WARNING: Non-finite values in {filename}")
    # Option 1: Interpolate
    # Option 2: Reject sequence
```

---

## 4.10 Key Takeaways

**What works:**
- ✅ Simple text parsing (no binary formats)
- ✅ Metadata encoded in filename (easy to extract)
- ✅ Quality indicators available (tracking state)
- ✅ All 567 sequences pass quality check

**Challenges:**
- ⚠️ Variable-length sequences (13-255 frames)
- ⚠️ Large dimensionality (60D per frame)
- ⚠️ Must handle missing/poor tracking

**Design choices:**
- Use confidence scores for quality filtering
- Flatten to (T, 60) for uniform representation
- Cross-subject split for realistic evaluation

---

**Next:** [05_NORMALIZATION.md](./05_NORMALIZATION.md) - StandardScaler normalization →

---

**Navigation:**
- [← 03_PIPELINE_OVERVIEW.md](./03_PIPELINE_OVERVIEW.md)
- [→ 05_NORMALIZATION.md](./05_NORMALIZATION.md)
- [↑ Index](./README.md)
