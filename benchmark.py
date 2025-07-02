import time
import numpy as np
import sys
import os

# Add the src directory to the Python path if that's where your modules are
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from loader import load_all_sequences
from classifier import classify_knn as classify_knn_classical
from q_classifier import classify_knn_quantum
from sklearn.model_selection import train_test_split

def run_benchmark():
    # Load data
    folder = "msr_action_data"
    sequences, labels = load_all_sequences(folder)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.3, random_state=42)
    
    # Classical DTW benchmark
    classical_start = time.time()
    classical_correct = 0
    for i, (test_seq, true_label) in enumerate(zip(X_test, y_test)):
        pred = classify_knn_classical(test_seq, X_train, y_train)
        if pred == true_label:
            classical_correct += 1
        print(f"Classical - Progress: {i+1}/{len(X_test)}", end="\r")
    
    classical_time = time.time() - classical_start
    classical_accuracy = classical_correct / len(y_test)
    
    # Quantum DTW benchmark
    quantum_start = time.time()
    quantum_correct = 0
    for i, (test_seq, true_label) in enumerate(zip(X_test, y_test)):
        pred = classify_knn_quantum(test_seq, X_train, y_train)
        if pred == true_label:
            quantum_correct += 1
        print(f"Quantum - Progress: {i+1}/{len(X_test)}", end="\r")
    
    quantum_time = time.time() - quantum_start
    quantum_accuracy = quantum_correct / len(y_test)
    
    # Print results
    print("\n\n===== BENCHMARK RESULTS =====")
    print(f"Classical DTW:")
    print(f"  - Accuracy: {classical_accuracy:.2%}")
    print(f"  - Time: {classical_time:.2f} seconds")
    print(f"\nQuantum DTW:")
    print(f"  - Accuracy: {quantum_accuracy:.2%}")
    print(f"  - Time: {quantum_time:.2f} seconds")
    print(f"\nAccuracy Improvement: {(quantum_accuracy-classical_accuracy):.2%}")
    print(f"Time Ratio: {quantum_time/classical_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()