import time
import cupy as cp
import numpy as np
import sys
import os
import math

# Add quantum_src FIRST so it takes precedence
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quantum_src'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Now import from quantum_src (after path is added)
from quantum_amp_est import (
    classify_knn_quantum_advanced, 
    classify_knn_quantum_hybrid,
    classify_knn_quantum_error_mitigated,
    adaptive_quantum_classifier
)
from loader import load_all_sequences
from gpu_classical_classifier import classify_knn_classical_gpu
from q_classifier import classify_knn_quantum_batch as classify_knn_quantum
from classifier import classify_knn_quantum_grover, grover_search_minimum_raw_unsafe
from qdtw import _qdtw_distance_scipy # Import the missing function
from sklearn.model_selection import train_test_split
def warmup_gpu():
    a = cp.ones((1000, 1000))
    b = cp.ones((1000, 1000))
    c = cp.matmul(a, b)
    cp.cuda.Stream.null.synchronize()
    print("GPU warmed up")

def run_comprehensive_quantum_benchmark():
    # Load data
    folder = "msr_action_data"
    sequences, labels = load_all_sequences(folder)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.3, random_state=42)
    
    print("="*80)
    print("COMPREHENSIVE QUANTUM ALGORITHM BENCHMARK")
    print("="*80)
    print(f"Dataset: {len(X_test)} test samples, {len(X_train)} training samples")
    print(f"Theoretical quantum advantage: O(√{len(X_train)}) = {math.sqrt(len(X_train)):.2f}x")
    print(f"\nQuantum Algorithms:")
    print("  1. Classical GPU (scipy optimized)")
    print("  2. Quantum DTW (scipy baseline)")  
    print("  3. Grover's Search (O(√N) database search)")
    print("  4. Quantum Amplitude Estimation (advanced optimization)")
    
    # GPU warmup and data loading
    print("\nWarming up GPU...")
    warmup_gpu()
    print("Loading training data to GPU...")
    X_train_gpu = [cp.array(seq, dtype=cp.float32) for seq in X_train]
    print("Training data loaded to GPU")
    
    # Store results
    results = {}
    
    # 1. Classical GPU DTW benchmark
    print("\n" + "="*60)
    print("1. CLASSICAL DTW (GPU) - BASELINE")
    print("="*60)
    classical_start = time.time()
    classical_correct = 0
    for i, (test_seq, true_label) in enumerate(zip(X_test, y_test)):
        pred = classify_knn_classical_gpu(test_seq, X_train_gpu, y_train)
        if pred == true_label:
            classical_correct += 1
        if (i + 1) % 10 == 0:
            print(f"Classical - Progress: {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")
    
    classical_time = time.time() - classical_start
    classical_accuracy = classical_correct / len(y_test)
    results['classical'] = {'time': classical_time, 'accuracy': classical_accuracy}
    print(f"✅ Classical completed: {classical_accuracy:.2%} accuracy in {classical_time:.2f}s")
    
    # 2. Quantum DTW benchmark (scipy baseline)
    print("\n" + "="*60)
    print("2. QUANTUM DTW (SCIPY) - SAME AS CLASSICAL")
    print("="*60)
    quantum_start = time.time()
    quantum_correct = 0
    for i, (test_seq, true_label) in enumerate(zip(X_test, y_test)):
        pred = classify_knn_quantum(test_seq, X_train_gpu, y_train)
        if pred == true_label:
            quantum_correct += 1
        if (i + 1) % 10 == 0:
            print(f"Quantum - Progress: {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")
    
    quantum_time = time.time() - quantum_start
    quantum_accuracy = quantum_correct / len(y_test)
    results['quantum'] = {'time': quantum_time, 'accuracy': quantum_accuracy}
    print(f"✅ Quantum DTW completed: {quantum_accuracy:.2%} accuracy in {quantum_time:.2f}s")
    
    # 3. Grover's Algorithm benchmark
    print("\n" + "="*60)
    print("3. GROVER'S QUANTUM SEARCH - O(√N) ADVANTAGE")
    print("="*60)
    print(f"Grover's provides √{len(X_train)} = {math.sqrt(len(X_train)):.2f}x theoretical speedup")
    grover_start = time.time()
    grover_correct = 0
    for i, (test_seq, true_label) in enumerate(zip(X_test, y_test)):
        pred = classify_knn_quantum_grover(test_seq, X_train_gpu, y_train)
        if pred == true_label:
            grover_correct += 1
        if (i + 1) % 10 == 0:
            print(f"Grover - Progress: {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")
    
    grover_time = time.time() - grover_start
    grover_accuracy = grover_correct / len(y_test)
    results['grover'] = {'time': grover_time, 'accuracy': grover_accuracy}
    print(f"✅ Grover's completed: {grover_accuracy:.2%} accuracy in {grover_time:.2f}s")
    
    # 4. Quantum Amplitude Estimation benchmark
    print("\n" + "="*60)
    print("4. QUANTUM AMPLITUDE ESTIMATION - ADVANCED QUANTUM")
    print("="*60)
    print("QAE uses Grover's as subroutine for optimization problems")
    amplitude_start = time.time()
    amplitude_correct = 0
    for i, (test_seq, true_label) in enumerate(zip(X_test, y_test)):
        pred = classify_knn_quantum_advanced(test_seq, X_train_gpu, y_train)
        if pred == true_label:
            amplitude_correct += 1
        if (i + 1) % 10 == 0:
            print(f"QAE - Progress: {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")
    
    amplitude_time = time.time() - amplitude_start
    amplitude_accuracy = amplitude_correct / len(y_test)
    results['amplitude'] = {'time': amplitude_time, 'accuracy': amplitude_accuracy}
    print(f"✅ QAE completed: {amplitude_accuracy:.2%} accuracy in {amplitude_time:.2f}s")
    
    # Comprehensive results analysis
    print("\n" + "="*80)
    print("COMPREHENSIVE QUANTUM ALGORITHM ANALYSIS")
    print("="*80)
    
    print("\n" + "="*60)
    print("5. HYBRID QUANTUM-CLASSICAL - INTELLIGENT SELECTION")
    print("="*60)
    print("Uses quantum only when classical is ambiguous")
    hybrid_start = time.time()
    hybrid_correct = 0
    for i, (test_seq, true_label) in enumerate(zip(X_test, y_test)):
        pred = classify_knn_quantum_hybrid(test_seq, X_train_gpu, y_train)
        if pred == true_label:
            hybrid_correct += 1
        if (i + 1) % 10 == 0:
            print(f"Hybrid - Progress: {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")
    
    hybrid_time = time.time() - hybrid_start
    hybrid_accuracy = hybrid_correct / len(y_test)
    results['hybrid'] = {'time': hybrid_time, 'accuracy': hybrid_accuracy}
    print(f"✅ Hybrid completed: {hybrid_accuracy:.2%} accuracy in {hybrid_time:.2f}s")
    

    # Add to grover_benchmark.py after the hybrid section

    # 6. Error-Mitigated Quantum benchmark
    print("\n" + "="*60)
    print("6. ERROR-MITIGATED QUANTUM - MULTIPLE SHOTS")
    print("="*60)
    print("Uses quantum consensus from multiple trials")
    error_mitigated_start = time.time()
    error_mitigated_correct = 0
    for i, (test_seq, true_label) in enumerate(zip(X_test, y_test)):
        pred = classify_knn_quantum_error_mitigated(test_seq, X_train_gpu, y_train)
        if pred == true_label:
            error_mitigated_correct += 1
        if (i + 1) % 10 == 0:
            print(f"Error-Mitigated - Progress: {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")
    
    error_mitigated_time = time.time() - error_mitigated_start
    error_mitigated_accuracy = error_mitigated_correct / len(y_test)
    results['error_mitigated'] = {'time': error_mitigated_time, 'accuracy': error_mitigated_accuracy}
    print(f"✅ Error-Mitigated completed: {error_mitigated_accuracy:.2%} accuracy in {error_mitigated_time:.2f}s")
    
    # 7. Adaptive Quantum benchmark
    print("\n" + "="*60)
    print("7. ADAPTIVE QUANTUM - INTELLIGENT ALGORITHM SELECTION")
    print("="*60)
    print("Chooses best quantum algorithm based on data characteristics")
    adaptive_start = time.time()
    adaptive_correct = 0
    for i, (test_seq, true_label) in enumerate(zip(X_test, y_test)):
        pred = adaptive_quantum_classifier(test_seq, X_train_gpu, y_train)
        if pred == true_label:
            adaptive_correct += 1
        if (i + 1) % 10 == 0:
            print(f"Adaptive - Progress: {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")
    
    adaptive_time = time.time() - adaptive_start
    adaptive_accuracy = adaptive_correct / len(y_test)
    results['adaptive'] = {'time': adaptive_time, 'accuracy': adaptive_accuracy}
    print(f"✅ Adaptive completed: {adaptive_accuracy:.2%} accuracy in {adaptive_time:.2f}s")



    print("\n" + "="*60)
    print("8. RAW GROVER'S SEARCH - UNSAFE/REALISTIC")
    print("="*60)
    print("Returns raw quantum measurement without classical fallback")
    
    def classify_knn_quantum_grover_raw(test_seq, train_seqs_gpu, train_labels, k=1):
        test_seq_gpu = cp.array(test_seq, dtype=cp.float32)
        distances = cp.array([_qdtw_distance_scipy(test_seq_gpu, ts_gpu) for ts_gpu in train_seqs_gpu])
        distances_cpu = distances.get()
        min_index = grover_search_minimum_raw_unsafe(distances_cpu)
        return train_labels[min_index]

    raw_grover_start = time.time()
    raw_grover_correct = 0
    for i, (test_seq, true_label) in enumerate(zip(X_test, y_test)):
        pred = classify_knn_quantum_grover_raw(test_seq, X_train_gpu, y_train)
        if pred == true_label:
            raw_grover_correct += 1
        if (i + 1) % 10 == 0:
            print(f"Raw Grover - Progress: {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")
    
    raw_grover_time = time.time() - raw_grover_start
    raw_grover_accuracy = raw_grover_correct / len(y_test)
    results['raw_grover'] = {'time': raw_grover_time, 'accuracy': raw_grover_accuracy}
    print(f"✅ Raw Grover completed: {raw_grover_accuracy:.2%} accuracy in {raw_grover_time:.2f}s")

    
    # Update the methods list
    methods = [
        ('Classical DTW (GPU)', 'classical', 'O(N)'),
        ('Quantum DTW (Scipy)', 'quantum', 'O(N)'),
        ('Grover Search', 'grover', 'O(√N)'),
        ('Amplitude Estimation', 'amplitude', 'O(√N)'),
        ('Hybrid Quantum', 'hybrid', 'O(√N) + Intelligence'),
        ('Error-Mitigated Quantum', 'error_mitigated', 'O(√N) + Error Correction'),
        ('Adaptive Quantum', 'adaptive', 'O(√N) + AI Selection'),
        ('Raw Grover Search', 'raw_grover', 'O(√N) Unsafe')
    ]
    
    
    print(f"\n{'Algorithm':<25} {'Complexity':<12} {'Accuracy':<12} {'Time':<12} {'Speedup':<10}")
    print("-" * 80)
    
    base_time = results['classical']['time']
    for name, key, complexity in methods:
        # Handle case where a new method might not be in results yet
        if key in results:
            acc = results[key]['accuracy']
            time_val = results[key]['time']
            speedup = base_time / time_val
            print(f"{name:<25} {complexity:<12} {acc:>10.2%} {time_val:>10.2f}s {speedup:>8.2f}x")
    
    # Quantum advantage analysis
    print(f"\n{'QUANTUM ADVANTAGE ANALYSIS'}")
    print("-" * 50)
    print(f"Dataset size: {len(X_train)} training samples")
    print(f"Theoretical √N advantage: {math.sqrt(len(X_train)):.2f}x")
    print(f"Grover's measured advantage: {base_time/results['grover']['time']:.2f}x")
    print(f"QAE measured advantage: {base_time/results['amplitude']['time']:.2f}x")
    
    # Algorithm relationship explanation
    print(f"\n{'ALGORITHM RELATIONSHIPS'}")
    print("-" * 40)
    print("Classical DTW: Point-by-point euclidean distance")
    print("Quantum DTW: Same as classical (scipy optimization)")
    print("Grover's: Quantum search for minimum distance index")
    print("QAE: Uses Grover's operator for amplitude estimation")
    print("→ QAE builds upon Grover's for optimization problems")
    
    # Accuracy analysis
    print(f"\n{'ACCURACY ANALYSIS'}")
    print("-" * 30)
    for name, key, _ in methods:
        acc = results[key]['accuracy']
        if acc < 0.8:
            status = "❌ Poor"
        elif acc < 0.85:
            status = "⚠️  Fair"
        else:
            status = "✅ Good"
        print(f"{name:<22} {acc:>8.2%} {status}")
    
    # Performance ranking
    times = [(results[key]['time'], name) for name, key, _ in methods]
    times.sort()
    
    print(f"\n{'PERFORMANCE RANKING (Speed)'}")
    print("-" * 35)
    for i, (time_val, name) in enumerate(times, 1):
        print(f"{i}. {name:<22} {time_val:>8.2f}s")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_quantum_benchmark()