"""
By-Class Analysis Experiment
=============================

Computes per-class accuracy for PCA vs VQD to identify which action
classes benefit most from quantum dimensionality reduction.

Provides interpretability: "VQD helps more for temporal-heavy actions"
or "VQD struggles with static poses" etc.

Author: VQD-DTW Research Team
Date: November 24, 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

# Import our modules
from archive.src.loader import load_all_sequences
from quantum.vqd_pca import vqd_quantum_pca
from dtw.dtw_runner import one_nn

# MSR Action3D class names
CLASS_NAMES = [
    "High arm wave",      # 0
    "Horizontal wave",    # 1
    "Hammer",             # 2
    "Hand catch",         # 3
    "Forward punch",      # 4
    "High throw",         # 5
    "Draw X",             # 6
    "Draw tick",          # 7
    "Draw circle (CCW)",  # 8
    "Hand wave",          # 9
    "Draw X",             # 10 (duplicate?)
    "Draw circle (CW)",   # 11
    "Hand clap",          # 12
    "Two hand wave",      # 13
    "Side boxing",        # 14
    "Bend",               # 15
    "Forward kick",       # 16
    "Side kick",          # 17
    "Jogging",            # 18
    "Tennis swing"        # 19
]

class ByClassExperiment:
    """Analyze VQD vs PCA performance by action class."""
    
    def __init__(self, k=8, seed=42, n_train=300, n_test=60, pre_k=16):
        self.k = k
        self.seed = seed
        self.n_train = n_train
        self.n_test = n_test
        self.pre_k = pre_k
        
        self.results = {
            'config': {
                'k': k,
                'seed': seed,
                'n_train': n_train,
                'n_test': n_test,
                'pre_k': pre_k,
                'date': datetime.now().isoformat()
            },
            'overall': {},      # Overall accuracy
            'by_class': {},     # Per-class metrics
            'top_vqd_gains': [],  # Classes where VQD wins most
            'top_pca_gains': []   # Classes where PCA wins most
        }
    
    def load_and_prepare_data(self):
        """Load data and split."""
        print(f"\n{'='*70}")
        print(f"Loading data with seed={self.seed}")
        print(f"{'='*70}")
        
        # Data is in parent directory
        data_path = Path(__file__).parent.parent / "msr_action_data"
        sequences, labels = load_all_sequences(str(data_path))
        print(f"Loaded {len(sequences)} sequences")
        
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            train_size=self.n_train,
            test_size=self.n_test,
            random_state=self.seed,
            stratify=labels
        )
        
        # Count samples per class
        unique, counts = np.unique(y_test, return_counts=True)
        print(f"\nTest set class distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"  Class {cls} ({CLASS_NAMES[cls]}): {cnt} samples")
        
        return X_train, X_test, y_train, y_test
    
    def build_frame_bank(self, X_train):
        """Build and pre-reduce frame bank."""
        print("\nBuilding frame bank...")
        frame_bank = np.vstack([seq for seq in X_train])
        
        self.scaler = StandardScaler()
        frame_bank_scaled = self.scaler.fit_transform(frame_bank)
        
        self.pca_pre = PCA(n_components=self.pre_k)
        frame_bank_reduced = self.pca_pre.fit_transform(frame_bank_scaled)
        print(f"Frame bank shape: {frame_bank_reduced.shape}")
        
        return frame_bank_reduced
    
    def project_sequence(self, seq, U_proj):
        """Project sequence with per-sequence centering."""
        seq_norm = self.scaler.transform(seq)
        seq_reduced = self.pca_pre.transform(seq_norm)
        mean = np.mean(seq_reduced, axis=0)
        seq_centered = seq_reduced - mean
        seq_proj = seq_centered @ U_proj.T
        return seq_proj
    
    def evaluate_method(self, X_train, X_test, y_train, y_test, U_proj, method_name):
        """Evaluate and return per-class accuracies."""
        print(f"\n--- {method_name} k={self.k} ---")
        
        # Project
        X_train_proj = [self.project_sequence(seq, U_proj) for seq in X_train]
        X_test_proj = [self.project_sequence(seq, U_proj) for seq in X_test]
        
        # DTW 1-NN - loop through test sequences
        start = time.time()
        y_pred = []
        y_train_arr = np.array(y_train)
        
        for test_seq in X_test_proj:
            pred, _ = one_nn(X_train_proj, y_train_arr, test_seq)
            y_pred.append(pred)
        
        y_pred = np.array(y_pred)
        elapsed = time.time() - start
        
        # Convert to numpy arrays
        y_test_arr = np.array(y_test)
        
        # Overall accuracy
        overall_acc = np.mean(y_pred == y_test_arr)
        print(f"{method_name} Overall Accuracy: {overall_acc*100:.1f}%")
        
        # Per-class accuracy
        per_class_acc = {}
        per_class_counts = {}
        
        for cls in range(20):
            # Find indices for this class
            mask = y_test_arr == cls
            if mask.sum() == 0:
                continue
            
            # Compute accuracy for this class
            class_acc = np.mean(y_pred[mask] == y_test_arr[mask])
            per_class_acc[cls] = class_acc
            per_class_counts[cls] = int(mask.sum())
            
            print(f"  Class {cls:2d} ({CLASS_NAMES[cls]:20s}): "
                  f"{class_acc*100:5.1f}% ({mask.sum()} samples)")
        
        return {
            'overall_accuracy': overall_acc,
            'time': elapsed,
            'per_class_accuracy': per_class_acc,
            'per_class_counts': per_class_counts
        }
    
    def run(self):
        """Run by-class analysis."""
        print("="*70)
        print("BY-CLASS ANALYSIS EXPERIMENT")
        print("="*70)
        print(f"k={self.k}, seed={self.seed}")
        print("="*70)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        frame_bank_reduced = self.build_frame_bank(X_train)
        
        # ========== PCA ==========
        print("\n### PCA ###")
        pca = PCA(n_components=self.k)
        pca.fit(frame_bank_reduced)
        U_pca = pca.components_
        
        pca_metrics = self.evaluate_method(X_train, X_test, y_train, y_test, 
                                          U_pca, "PCA")
        
        # ========== VQD ==========
        print("\n### VQD ###")
        U_vqd, eigenvalues_vqd, logs = vqd_quantum_pca(
            frame_bank_reduced,
            n_components=self.k,
            num_qubits=4,
            max_depth=2,
            penalty_scale='auto',
            ramped_penalties=True,
            entanglement='alternating',
            maxiter=200,
            validate=True
        )
        
        if 'U_vqd_aligned' in logs:
            U_proj = logs['U_vqd_aligned']
        else:
            U_proj = U_vqd
        
        vqd_metrics = self.evaluate_method(X_train, X_test, y_train, y_test,
                                          U_proj, "VQD")
        
        # ========== Compute Deltas ==========
        print(f"\n{'='*70}")
        print("PER-CLASS VQD-PCA GAPS")
        print(f"{'='*70}")
        
        per_class_deltas = {}
        
        for cls in range(20):
            if cls in pca_metrics['per_class_accuracy'] and cls in vqd_metrics['per_class_accuracy']:
                pca_acc = pca_metrics['per_class_accuracy'][cls]
                vqd_acc = vqd_metrics['per_class_accuracy'][cls]
                delta = vqd_acc - pca_acc
                
                per_class_deltas[cls] = {
                    'pca_accuracy': pca_acc,
                    'vqd_accuracy': vqd_acc,
                    'delta': delta,
                    'count': pca_metrics['per_class_counts'][cls],
                    'class_name': CLASS_NAMES[cls]
                }
                
                print(f"Class {cls:2d} ({CLASS_NAMES[cls]:20s}): "
                      f"PCA={pca_acc*100:5.1f}%, VQD={vqd_acc*100:5.1f}%, "
                      f"Δ={delta*100:+6.1f}%")
        
        # Store results
        self.results['overall'] = {
            'pca': pca_metrics['overall_accuracy'],
            'vqd': vqd_metrics['overall_accuracy'],
            'delta': vqd_metrics['overall_accuracy'] - pca_metrics['overall_accuracy']
        }
        
        self.results['by_class'] = per_class_deltas
        
        # Identify top gains
        sorted_by_delta = sorted(per_class_deltas.items(), 
                                key=lambda x: x[1]['delta'], reverse=True)
        
        self.results['top_vqd_gains'] = [
            {'class': cls, **info} for cls, info in sorted_by_delta[:5]
        ]
        
        self.results['top_pca_gains'] = [
            {'class': cls, **info} for cls, info in sorted_by_delta[-5:]
        ]
        
        # Print top gains
        print(f"\n{'='*70}")
        print("TOP 5 CLASSES WHERE VQD WINS")
        print(f"{'='*70}")
        for item in self.results['top_vqd_gains']:
            print(f"Class {item['class']:2d} ({item['class_name']:20s}): "
                  f"{item['delta']*100:+6.1f}% (VQD={item['vqd_accuracy']*100:.1f}%, "
                  f"PCA={item['pca_accuracy']*100:.1f}%)")
        
        print(f"\n{'='*70}")
        print("TOP 5 CLASSES WHERE PCA WINS")
        print(f"{'='*70}")
        for item in self.results['top_pca_gains']:
            print(f"Class {item['class']:2d} ({item['class_name']:20s}): "
                  f"{item['delta']*100:+6.1f}% (VQD={item['vqd_accuracy']*100:.1f}%, "
                  f"PCA={item['pca_accuracy']*100:.1f}%)")
        
        # Save results
        self.save_results()
        
        # Create visualization
        self.create_visualization(per_class_deltas)
        
        print("\n" + "="*70)
        print("BY-CLASS ANALYSIS COMPLETE!")
        print(f"Results saved to: results/by_class_results.json")
        print(f"Figure saved to: figures/by_class_comparison.png")
        print("="*70)
    
    def create_visualization(self, per_class_deltas):
        """Create bar chart of VQD-PCA gaps by class."""
        print("\nCreating visualization...")
        
        # Sort by delta
        sorted_classes = sorted(per_class_deltas.items(), 
                              key=lambda x: x[1]['delta'])
        
        classes = [cls for cls, _ in sorted_classes]
        deltas = [info['delta'] * 100 for _, info in sorted_classes]
        names = [CLASS_NAMES[cls] for cls in classes]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color: green for VQD wins, red for PCA wins
        colors = ['green' if d > 0 else 'red' for d in deltas]
        
        # Horizontal bar chart
        y_pos = np.arange(len(classes))
        ax.barh(y_pos, deltas, color=colors, alpha=0.7)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('VQD - PCA Accuracy (%)', fontsize=12)
        ax.set_title(f'Per-Class VQD Advantage (k={self.k}, seed={self.seed})', 
                    fontsize=14, fontweight='bold')
        
        # Zero line
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        
        # Grid
        ax.grid(axis='x', alpha=0.3)
        
        # Annotate values
        for i, (cls, delta) in enumerate(zip(classes, deltas)):
            ax.text(delta + 0.5 if delta > 0 else delta - 0.5, i, 
                   f'{delta:+.1f}%', 
                   va='center', ha='left' if delta > 0 else 'right',
                   fontsize=9)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(__file__).parent / "figures" / "by_class_comparison.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
        
        plt.close()
    
    def save_results(self):
        """Save results to JSON."""
        output_path = Path(__file__).parent / "results" / "by_class_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    experiment = ByClassExperiment(
        k=8,
        seed=42,
        n_train=300,
        n_test=60,
        pre_k=16
    )
    
    experiment.run()
