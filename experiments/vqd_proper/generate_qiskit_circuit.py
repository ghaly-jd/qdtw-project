"""
Generate Qiskit Circuit Diagram for VQD Ansatz
==============================================

Creates the actual Qiskit circuit visualization used in VQD experiments.

Author: VQD-DTW Research Team
Date: November 25, 2025
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

def create_vqd_ansatz(n_qubits=4, depth=2):
    """
    Create the VQD ansatz circuit exactly as used in experiments.
    
    Architecture:
    - RY rotation gates on all qubits
    - Linear entanglement (CNOT chain)
    - Repeated for 'depth' layers
    
    Parameters:
    -----------
    n_qubits : int
        Number of qubits (default: 4)
    depth : int
        Circuit depth (default: 2)
    
    Returns:
    --------
    QuantumCircuit : The parameterized ansatz
    """
    circuit = QuantumCircuit(n_qubits)
    
    param_idx = 0
    
    for layer in range(depth):
        # Rotation layer - RY gates on all qubits
        for qubit in range(n_qubits):
            theta = Parameter(f'θ_{param_idx}')
            circuit.ry(theta, qubit)
            param_idx += 1
        
        # Entanglement layer - CNOT chain
        for qubit in range(n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        
        # Add barrier for visualization clarity
        circuit.barrier()
    
    return circuit

def create_vqd_ansatz_with_values(n_qubits=4, depth=2):
    """Create ansatz with example parameter values for visualization."""
    circuit = QuantumCircuit(n_qubits)
    
    # Use some example parameter values
    example_params = [0.5, 1.2, -0.3, 0.8, 1.5, -0.7, 0.2, 1.1]
    param_idx = 0
    
    for layer in range(depth):
        # Rotation layer
        for qubit in range(n_qubits):
            circuit.ry(example_params[param_idx], qubit)
            param_idx += 1
        
        # Entanglement layer
        for qubit in range(n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        
        circuit.barrier()
    
    return circuit

if __name__ == "__main__":
    print("="*70)
    print("GENERATING QISKIT VQD ANSATZ CIRCUIT")
    print("="*70)
    
    # Create the parameterized ansatz
    print("\n1. Creating parameterized ansatz...")
    ansatz = create_vqd_ansatz(n_qubits=4, depth=2)
    
    print(f"   Circuit properties:")
    print(f"   - Qubits: {ansatz.num_qubits}")
    print(f"   - Parameters: {ansatz.num_parameters}")
    print(f"   - Depth: {ansatz.depth()}")
    print(f"   - Gates: {sum(ansatz.count_ops().values())}")
    
    # Save multiple styles
    print("\n2. Generating circuit diagrams...")
    
    # Style 1: Matplotlib style (best for presentations)
    print("   a) Matplotlib style...")
    fig = ansatz.draw(output='mpl', style='iqp', fold=-1)
    fig.savefig(FIG_DIR / 'vqd_circuit_qiskit.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"      ✓ Saved: {FIG_DIR / 'vqd_circuit_qiskit.png'}")
    
    # Style 2: Text representation
    print("   b) Text representation...")
    text_circuit = ansatz.draw(output='text')
    with open(FIG_DIR / 'vqd_circuit_text.txt', 'w') as f:
        f.write(str(text_circuit))
    print(f"      ✓ Saved: {FIG_DIR / 'vqd_circuit_text.txt'}")
    
    # Style 3: LaTeX source (for papers)
    print("   c) LaTeX source...")
    latex_circuit = ansatz.draw(output='latex_source')
    with open(FIG_DIR / 'vqd_circuit_latex.txt', 'w') as f:
        f.write(latex_circuit)
    print(f"      ✓ Saved: {FIG_DIR / 'vqd_circuit_latex.txt'}")
    
    # Create version with example values
    print("\n3. Creating circuit with example parameter values...")
    ansatz_with_values = create_vqd_ansatz_with_values(n_qubits=4, depth=2)
    fig2 = ansatz_with_values.draw(output='mpl', style='iqp', fold=-1)
    fig2.savefig(FIG_DIR / 'vqd_circuit_with_values.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"   ✓ Saved: {FIG_DIR / 'vqd_circuit_with_values.png'}")
    
    # Print text version to console
    print("\n" + "="*70)
    print("TEXT REPRESENTATION:")
    print("="*70)
    print(ansatz.draw(output='text'))
    
    print("\n" + "="*70)
    print("CIRCUIT STATISTICS:")
    print("="*70)
    print(f"Total qubits: {ansatz.num_qubits}")
    print(f"Total parameters: {ansatz.num_parameters}")
    print(f"Circuit depth: {ansatz.depth()}")
    print(f"Gate counts: {dict(ansatz.count_ops())}")
    
    print("\n" + "="*70)
    print("FILES GENERATED:")
    print("="*70)
    print(f"1. vqd_circuit_qiskit.png - High-res circuit diagram (MAIN)")
    print(f"2. vqd_circuit_with_values.png - Circuit with example values")
    print(f"3. vqd_circuit_text.txt - Text representation")
    print(f"4. vqd_circuit_latex.txt - LaTeX source code")
    print(f"\nAll saved to: {FIG_DIR}/")
    print("\n✨ Qiskit circuit ready for presentation! ✨")
