import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

# Add project directories to import your code
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'quantum_src'))

# --- Helper Functions & Setup ---

def setup_visualization_dir():
    """Creates the output directory for all visuals."""
    VIZ_DIR = "visualizations"
    if not os.path.exists(VIZ_DIR):
        os.makedirs(VIZ_DIR)
    print(f"Visualizations will be saved in: {os.path.abspath(VIZ_DIR)}")
    return VIZ_DIR

# --- Visualization Functions ---

def visualize_skeleton_pose(save_path):
    """
    VISUAL 1: A static 3D plot of a single skeleton pose.
    This introduces the dataset.
    """
    from loader import load_skeleton_file
    print("Generating skeleton pose visual...")

    # Kinect 20-joint edges
    EDGES = [(0, 1), (1, 2), (2, 3), (1, 5), (5, 6), (6, 7), (1, 9), (9, 10), (10, 11), (1, 12), (12, 13), (13, 14), (1, 16), (16, 17), (17, 18)]
    
    seq = load_skeleton_file("msr_action_data/a08_s02_e01_skeleton.txt") # "High throw" action
    frame = seq[15] # A representative frame

    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(frame[:, 0], frame[:, 2], frame[:, 1], c='blue', s=50, depthshade=True)
    for i, j in EDGES:
        ax.plot([frame[i, 0], frame[j, 0]], [frame[i, 2], frame[j, 2]], [frame[i, 1], frame[j, 1]], 'r-', linewidth=3)

    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z (Height)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Y (Depth)', fontsize=12, fontweight='bold')
    ax.set_title('Sample Skeleton Pose', fontsize=16, fontweight='bold')
    ax.view_init(elev=20, azim=35)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved to {save_path}")

def visualize_skeleton_animation(save_path):
    """
    VISUAL 2: An animated GIF of a skeleton performing an action.
    This shows the time-series nature of the data.
    """
    from loader import load_skeleton_file
    print("Generating skeleton animation (this may take a moment)...")

    EDGES = [(0, 1), (1, 2), (2, 3), (1, 5), (5, 6), (6, 7), (1, 9), (9, 10), (10, 11), (1, 12), (12, 13), (13, 14), (1, 16), (16, 17), (17, 18)]
    seq = load_skeleton_file("msr_action_data/a01_s01_e01_skeleton.txt") # "High wave" action

    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        frame = seq[frame_idx]
        ax.scatter(frame[:, 0], frame[:, 2], frame[:, 1], c='blue', s=50, depthshade=True)
        for i, j in EDGES:
            ax.plot([frame[i, 0], frame[j, 0]], [frame[i, 2], frame[j, 2]], [frame[i, 1], frame[j, 1]], 'r-', linewidth=3)
        ax.set_xlabel('X'); ax.set_ylabel('Z (Height)'); ax.set_zlabel('Y (Depth)')
        ax.set_title(f'Action: High Wave (Frame {frame_idx+1})', fontsize=16, fontweight='bold')
        ax.view_init(elev=20, azim=35)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        return fig,

    # Create animation and save as GIF
    anim = FuncAnimation(fig, update, frames=len(seq), interval=100, blit=False)
    anim.save(save_path, writer='imageio', fps=10)
    plt.close()
    print(f"  -> Saved to {save_path}")

def visualize_dtw_alignment(save_path):
    """
    VISUAL 3: The DTW cost matrix and alignment path.
    This explains the classical algorithm.
    """
    from dtw import dtw
    from scipy.spatial.distance import cdist
    from loader import load_skeleton_file
    from sklearn.decomposition import PCA
    print("Generating DTW alignment visual...")

    # Load two different "high wave" actions
    seq1 = load_skeleton_file("msr_action_data/a01_s01_e01_skeleton.txt")
    seq2 = load_skeleton_file("msr_action_data/a01_s02_e01_skeleton.txt")

    # Use PCA to reduce each 60-dim frame to 1-dim for clear visualization
    pca = PCA(n_components=1)
    x = pca.fit_transform(np.vstack([seq1.reshape(len(seq1), -1), seq2.reshape(len(seq2), -1)]))
    x1 = x[:len(seq1)]
    x2 = x[len(seq1):]

    # Get DTW alignment
    alignment = dtw(x1, x2, keep_internals=True)

    # --- FIXED PLOTTING LOGIC ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the cost matrix (the heatmap)
    ax.imshow(alignment.costMatrix.T, origin='lower', cmap='viridis', interpolation='nearest')
    
    # Plot the optimal path on top of the heatmap
    ax.plot(alignment.index1, alignment.index2, color='red', linewidth=3, label='Optimal Path')
    
    ax.set_title("DTW Cost Matrix & Optimal Path", fontsize=16, fontweight='bold')
    ax.set_xlabel("Action 1 Frames", fontsize=12)
    ax.set_ylabel("Action 2 Frames", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved to {save_path}")

def visualize_grover_circuit(save_path):
    """
    VISUAL 4: A diagram of the Grover's algorithm circuit.
    This explains the core quantum algorithm.
    """
    from qiskit import QuantumCircuit
    print("Generating Grover's circuit visual...")

    n = 3 # Example with 3 qubits
    oracle = QuantumCircuit(n, name='Oracle')
    oracle.cz(0, 2) # Example oracle marking '101'
    
    diffuser = QuantumCircuit(n, name='Diffuser')
    diffuser.h(range(n)); diffuser.x(range(n)); diffuser.h(n-1)
    diffuser.mcx(list(range(n-1)), n-1)
    diffuser.h(n-1); diffuser.x(range(n)); diffuser.h(range(n))

    qc = QuantumCircuit(n, name="Grover Iteration")
    qc.h(range(n))
    qc.append(oracle, range(n))
    qc.append(diffuser, range(n))
    
    fig = qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
    fig.suptitle("Grover's Algorithm: One Iteration", fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved to {save_path}")

def visualize_results_chart(save_path):
    """
    VISUAL 5: A bar chart of the final benchmark performance.
    This summarizes the key findings.
    """
    print("Generating final results bar chart...")
    
    # Data from your final benchmark run
    results = {
        'Classical GPU': 94.94,
        'Quantum DTW': 96.57,
        'Hybrid': 97.17,
        'QAE': 97.22,
        'Error-Mitigated': 97.40,
        'Adaptive': 97.44,
        'Grover': 116.15
    }
    
    names = list(results.keys())
    times = list(results.values())
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#1f77b4' if 'Classical' in name else '#ff7f0e' if 'Grover' in name else '#2ca02c' for name in names]
    bars = ax.bar(names, times, color=colors)
    
    ax.set_ylabel('Total Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Algorithm Performance Benchmark (Lower is Better)', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    ax.bar_label(bars, fmt='%.2f s', padding=3, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved to {save_path}")



def visualize_qae_fixed_circuit(save_path):
    """
    VISUAL 6: The circuit from the first QAE implementation.
    This shows a phase-encoding approach.
    """
    from qiskit import QuantumCircuit
    import math
    print("Generating 'QAE Fixed' circuit visual...")

    num_qubits = 3
    ancilla = num_qubits # The last qubit is the ancilla

    qc = QuantumCircuit(num_qubits + 1, name="QAE (Phase Encoding)")
    qc.h(range(num_qubits))
    qc.barrier()

    # Represent the phase encoding for one state (e.g., |101>)
    # This is a conceptual representation
    qc.x(0) # To target state |101|
    qc.rz(math.pi * 0.8, 2) # Apply phase based on amplitude
    qc.x(0)
    qc.barrier()

    # Entanglement with ancilla
    for i in range(num_qubits):
        qc.cx(i, ancilla)  # FIXED: Changed cnot to cx
    
    fig = qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
    fig.suptitle("QAE Circuit 1: Phase Encoding", fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved to {save_path}")
def visualize_qae_enhanced_circuit(save_path):
    """
    VISUAL 7: The circuit from the 'enhanced' QAE implementation.
    This shows a simplified, targeted Grover search.
    """
    from qiskit import QuantumCircuit
    print("Generating 'QAE Enhanced' circuit visual...")

    num_qubits = 4
    qc = QuantumCircuit(num_qubits, name="QAE (Targeted Search)")
    qc.h(range(num_qubits))
    qc.barrier()

    # Represent marking a "good" state
    qc.x([0, 2]) # To target a state like |1010|
    
    # --- FIXED: Replaced mcz with H-MCX-H ---
    # This is the standard way to build a multi-controlled Z gate
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    # --- End of fix ---
    
    qc.x([0, 2])
    qc.barrier()

    # Diffusion
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    
    # --- FIXED: Replaced mcz with H-MCX-H ---
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    # --- End of fix ---
    
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))

    fig = qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
    fig.suptitle("QAE Circuit 2: Targeted Grover Search", fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved to {save_path}")

# --- Main Execution ---
if __name__ == "__main__":
    VIZ_DIR = setup_visualization_dir()

    # Generate all visuals one by one
    visualize_skeleton_pose(os.path.join(VIZ_DIR, "1_skeleton_pose.png"))
    visualize_skeleton_animation(os.path.join(VIZ_DIR, "2_skeleton_animation.gif"))
    visualize_dtw_alignment(os.path.join(VIZ_DIR, "3_dtw_alignment.png"))
    visualize_grover_circuit(os.path.join(VIZ_DIR, "4_grover_circuit.png"))
    visualize_results_chart(os.path.join(VIZ_DIR, "5_results_performance_chart.png"))
    
    # Add the new circuit visualizations
    visualize_qae_fixed_circuit(os.path.join(VIZ_DIR, "6_qae_fixed_circuit.png"))
    visualize_qae_enhanced_circuit(os.path.join(VIZ_DIR, "7_qae_enhanced_circuit.png"))

    print("\nAll visualizations have been generated successfully!")