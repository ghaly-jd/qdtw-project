"""
Quantum Principal Component Analysis (qPCA) Simulator.

This module provides a simulation of quantum PCA that:
1. Constructs an empirical density matrix from normalized data
2. Extracts top-k eigenvectors (principal components)
3. Includes quantum circuit scaffolds to document the quantum steps

The quantum approach uses density matrix diagonalization, which in a
real quantum implementation would use quantum phase estimation on a
state preparation circuit.
"""

import logging
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def qpca_directions(X: np.ndarray, k: int, auto_normalize: bool = True) -> np.ndarray:
    """
    Compute top-k principal components via density matrix diagonalization.

    This simulates quantum PCA by:
    1. Building empirical density matrix: rho = (1/M) * sum_i |x_i><x_i|
    2. Computing top-k eigenvectors of rho classically
    3. Returning eigenvectors as principal directions

    In a real quantum implementation, this would use:
    - Quantum state preparation for each |x_i>
    - Quantum phase estimation to extract eigenvalues/eigenvectors
    - Measurement to recover classical results

    Args:
        X: Input data matrix of shape [M, D]
           If auto_normalize=True, rows will be L2-normalized automatically
           If auto_normalize=False, rows must already be unit vectors
        k: Number of principal components to extract (k <= min(M, D))
        auto_normalize: If True, normalize rows to unit vectors (default: True)

    Returns:
        U: Principal components matrix of shape [D, k]
           Columns are orthonormal eigenvectors of the density matrix,
           ordered by decreasing eigenvalue (variance)

    Raises:
        ValueError: If X is not 2D, k is invalid, or rows are not normalized

    Example:
        >>> X = np.random.randn(1000, 60)
        >>> U = qpca_directions(X, k=10)  # Auto-normalizes
        >>> print(U.shape)  # (60, 10)
    """
    # Validate inputs
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")

    M, D = X.shape

    if k <= 0 or k > min(M, D):
        raise ValueError(
            f"k must be in range [1, min(M, D)] = [1, {min(M, D)}], got {k}"
        )

    # Auto-normalize rows if requested (for standardized or unnormalized data)
    if auto_normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        X = X / norms
        logger.info(f"Auto-normalized {M} vectors to unit length")
    else:
        # Verify rows are normalized (unit vectors)
        norms = np.linalg.norm(X, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            max_deviation = np.max(np.abs(norms - 1.0))
            raise ValueError(
                f"All rows of X must be unit vectors. "
                f"Max deviation from norm=1: {max_deviation:.2e}"
            )

    logger.info(f"Building density matrix from {M} normalized vectors in {D}D space")

    # Step 1: Build empirical density matrix
    # rho = (1/M) * sum_i |x_i><x_i| = (1/M) * X.T @ X
    rho = (X.T @ X) / M

    logger.info(f"Density matrix shape: {rho.shape}")
    logger.info(f"Density matrix trace: {np.trace(rho):.6f} (should be ≈1)")

    # Verify density matrix properties
    trace_val = np.trace(rho)
    if not np.isclose(trace_val, 1.0, atol=1e-3):
        logger.warning(
            f"Density matrix trace = {trace_val:.6f}, expected ≈1.0"
        )

    # Check if Hermitian (should be for real data)
    is_hermitian = np.allclose(rho, rho.T, atol=1e-10)
    logger.info(f"Density matrix is Hermitian: {is_hermitian}")

    # Step 2: Compute eigenvectors and eigenvalues
    # Use eigh since rho is Hermitian (symmetric for real data)
    logger.info("Diagonalizing density matrix...")
    eigenvalues, eigenvectors = np.linalg.eigh(rho)

    # eigh returns eigenvalues in ascending order, we want descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    logger.info(f"Top 5 eigenvalues: {eigenvalues[:5]}")
    logger.info(f"Sum of all eigenvalues: {np.sum(eigenvalues):.6f}")

    # Step 3: Extract top-k eigenvectors
    U = eigenvectors[:, :k]

    # Verify orthonormality
    ortho_check = U.T @ U
    identity = np.eye(k)
    max_deviation = np.max(np.abs(ortho_check - identity))
    logger.info(f"Orthonormality check: max deviation = {max_deviation:.2e}")

    # Log quantum implementation details
    _log_quantum_parameters(M, D, k)

    return U


def _log_quantum_parameters(M: int, D: int, k: int) -> None:
    """
    Log conceptual quantum implementation parameters.

    In a real quantum implementation, these parameters would be used for:
    - State preparation circuits
    - Quantum phase estimation (QPE)
    - Measurement and post-processing

    Args:
        M: Number of data samples
        D: Dimensionality of each sample
        k: Number of principal components to extract
    """
    # Pad to power of 2 for quantum state representation
    n_qubits = int(np.ceil(np.log2(D)))
    padded_dim = 2 ** n_qubits

    logger.info("=" * 70)
    logger.info("QUANTUM IMPLEMENTATION PARAMETERS (Conceptual)")
    logger.info("=" * 70)
    logger.info(f"Data dimension D: {D}")
    logger.info(f"Padded dimension (2^n): {padded_dim}")
    logger.info(f"Number of qubits for state prep: {n_qubits}")
    logger.info(f"Number of samples M: {M}")
    logger.info(f"Number of components k: {k}")

    # Quantum Phase Estimation parameters
    precision_qubits = int(np.ceil(np.log2(k))) + 4  # Extra precision
    logger.info(f"QPE precision qubits: {precision_qubits}")

    # Estimated circuit depth (very rough estimate)
    # State prep: O(D), QPE: O(2^precision), Controlled operations: O(M)
    estimated_depth = padded_dim + (2 ** precision_qubits) + M
    logger.info(f"Estimated circuit depth: {estimated_depth}")

    # Shot count for measurement
    # Higher shots needed for accurate eigenvalue estimation
    shot_count = 8192 * k  # Scale with number of components
    logger.info(f"Recommended shot count: {shot_count}")

    logger.info("=" * 70)


def build_state_prep_circuit(x: np.ndarray) -> QuantumCircuit:
    """
    Build a Qiskit circuit to prepare quantum state |x> from classical vector.

    This circuit documents the quantum state preparation step needed for qPCA.
    The vector x (dimension D=60) is padded to dimension 64 (2^6) and encoded
    as amplitudes of a 6-qubit quantum state.

    Mathematical mapping:
        Classical vector x ∈ R^60 → Quantum state |ψ> ∈ C^64
        |ψ> = sum_{i=0}^{63} α_i |i>

    where:
        α_i = x[i] / ||x|| for i < 60  (normalized amplitudes)
        α_i = 0            for i >= 60 (padding)

    The circuit uses the 'initialize' instruction which decomposes the
    state preparation into a sequence of rotation and CNOT gates.

    Args:
        x: Classical vector of shape [D] or [D,], typically D=60
           Should be a unit vector (L2 norm = 1)

    Returns:
        qc: QuantumCircuit with 6 qubits that prepares state |x>
            The circuit includes:
            - Initialization from amplitudes
            - Proper normalization
            - Padding to 64 dimensions

    Example:
        >>> x = np.random.randn(60)
        >>> x = x / np.linalg.norm(x)  # Normalize
        >>> qc = build_state_prep_circuit(x)
        >>> print(qc.depth())
        >>> print(qc.num_qubits)  # 6
    """
    # Validate input
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")

    D = len(x)

    # Verify normalization
    norm = np.linalg.norm(x)
    if not np.isclose(norm, 1.0, atol=1e-6):
        logger.warning(
            f"Input vector not normalized (norm={norm:.6f}). "
            "Normalizing automatically."
        )
        x = x / norm

    # Pad to next power of 2 (64 for D=60)
    n_qubits = int(np.ceil(np.log2(D)))
    padded_dim = 2 ** n_qubits

    # Create padded amplitudes
    amplitudes = np.zeros(padded_dim, dtype=np.float64)
    amplitudes[:D] = x

    logger.info(f"State preparation: {D}D vector → {padded_dim}D quantum state")
    logger.info(f"Number of qubits: {n_qubits}")

    # Create quantum circuit
    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr)

    # Initialize state using amplitude encoding
    # This automatically decomposes into gates
    qc.initialize(amplitudes, qr)

    # Add metadata to circuit
    qc.name = f"StatePrep_D{D}_to_{padded_dim}"
    qc.metadata = {
        'original_dim': D,
        'padded_dim': padded_dim,
        'n_qubits': n_qubits,
        'method': 'amplitude_encoding'
    }

    logger.info(f"Circuit depth: {qc.depth()}")
    logger.info(f"Circuit gate count: {qc.size()}")

    return qc


def save_qpca_components(U: np.ndarray, k: int) -> str:
    """
    Save qPCA principal components to disk.

    Components are saved to: quantum/outputs/Uq_k{K}.npy

    Args:
        U: Principal components matrix of shape [D, k]
        k: Number of components (used in filename)

    Returns:
        path: Absolute path to saved file

    Example:
        >>> U = qpca_directions(X, k=10)
        >>> path = save_qpca_components(U, k=10)
        >>> print(path)  # quantum/outputs/Uq_k10.npy
    """
    # Get project root (parent of quantum directory)
    quantum_dir = Path(__file__).parent
    outputs_dir = quantum_dir / 'outputs'
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Save with qPCA-specific naming
    filename = f"Uq_k{k}.npy"
    filepath = outputs_dir / filename

    np.save(filepath, U)

    logger.info(f"Saved qPCA components to: {filepath}")
    logger.info(f"Shape: {U.shape}, k={k}")

    return str(filepath.absolute())


def load_qpca_components(k: int) -> np.ndarray:
    """
    Load qPCA principal components from disk.

    Loads from: quantum/outputs/Uq_k{K}.npy

    Args:
        k: Number of components (used in filename)

    Returns:
        U: Principal components matrix of shape [D, k]

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> U = load_qpca_components(k=10)
        >>> print(U.shape)  # (60, 10)
    """
    quantum_dir = Path(__file__).parent

    # Try multiple possible locations/formats for compatibility:
    # 1) results/Uq_k{k}_std.npz or results/Uq_k{k}.npz  (new CLI output)
    # 2) quantum/outputs/Uq_k{k}.npy                     (legacy save)
    candidate_paths = []

    # Prefer results/ (project-level outputs)
    candidate_paths.append(Path('results') / f'Uq_k{k}_std.npz')
    candidate_paths.append(Path('results') / f'Uq_k{k}.npz')

    # Legacy location under quantum/outputs
    candidate_paths.append(quantum_dir / 'outputs' / f'Uq_k{k}.npy')
    candidate_paths.append(quantum_dir / 'outputs' / f'Uq_k{k}.npz')

    for p in candidate_paths:
        if p.exists():
            logger.info(f"Found qPCA components file: {p}")
            # Load .npz that contains keys (U, explained_variance_ratio)
            if p.suffix == '.npz':
                data = np.load(p, allow_pickle=True)
                # Prefer key 'U' if present
                if 'U' in data:
                    U = data['U']
                else:
                    # If only a single array stored, try to load it
                    # (np.savez with unnamed arrays uses keys like 'arr_0')
                    try:
                        U = data['arr_0']
                    except Exception:
                        raise ValueError(f"Unable to extract 'U' from {p}")
                logger.info(f"Loaded qPCA components from: {p} (npz)")
                logger.info(f"Shape: {U.shape}")
                return U
            else:
                # .npy direct array
                U = np.load(p)
                logger.info(f"Loaded qPCA components from: {p} (npy)")
                logger.info(f"Shape: {U.shape}")
                return U

    # If none found, raise helpful error listing checked paths
    paths_str = '\n'.join(str(x) for x in candidate_paths)
    raise FileNotFoundError(
        f"qPCA components file not found. Checked the following paths:\n{paths_str}\n"
        f"Run `python quantum/qpca.py --frames data/frame_bank_std.npy --k {k}` to generate one."
    )


def compute_principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Compute principal angles between two subspaces.

    Principal angles measure the "distance" between two subspaces
    spanned by the columns of U1 and U2.

    The principal angles θ_i ∈ [0, π/2] satisfy:
        cos(θ_i) = σ_i
    where σ_i are the singular values of U1.T @ U2.

    Args:
        U1: Orthonormal matrix of shape [D, k1]
        U2: Orthonormal matrix of shape [D, k2]

    Returns:
        angles: Principal angles in degrees, shape [min(k1, k2)]
                Sorted in ascending order

    Example:
        >>> U1 = classical_pca(X, k=10)[0]
        >>> U2 = qpca_directions(X, k=10)
        >>> angles = compute_principal_angles(U1, U2)
        >>> print(f"Max angle: {angles[-1]:.2f}°")
    """
    # Compute SVD of U1.T @ U2
    # Singular values are cosines of principal angles
    _, sigma, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)

    # Clamp to [0, 1] to avoid numerical issues with arccos
    sigma = np.clip(sigma, 0.0, 1.0)

    # Convert to angles in degrees
    angles_rad = np.arccos(sigma)
    angles_deg = np.degrees(angles_rad)

    return angles_deg


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute Quantum PCA (qPCA) on frame bank'
    )
    parser.add_argument(
        '--frames',
        type=str,
        required=True,
        help='Path to frame bank .npy file (e.g., data/frame_bank_std.npy)'
    )
    parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='Number of principal components to compute'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for Uq file (default: results/Uq_k{K}_std.npz)'
    )
    
    args = parser.parse_args()
    
    # Load frame bank
    print(f"Loading frame bank from {args.frames}...")
    X = np.load(args.frames)
    print(f"Frame bank shape: {X.shape}")
    
    # Compute quantum PCA (auto-normalizes standardized data)
    print(f"\nComputing quantum PCA with k={args.k}...")
    U = qpca_directions(X, k=args.k, auto_normalize=True)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Use _std suffix if input has it
        suffix = '_std' if '_std' in args.frames else ''
        output_path = Path(f'results/Uq_k{args.k}{suffix}.npz')
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute explained variance (eigenvalues of density matrix)
    # For quantum PCA, we need to recompute the density matrix
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms == 0, 1.0, norms)
    rho = (X_norm.T @ X_norm) / X.shape[0]
    eigenvalues = np.linalg.eigvalsh(rho)[::-1]  # Descending order
    
    # Compute explained variance ratios
    total_var = np.sum(eigenvalues)
    evr = eigenvalues[:args.k] / total_var
    
    # Save components with explained variance
    np.savez(
        output_path,
        U=U,
        explained_variance_ratio=evr
    )
    
    print(f"\n✅ Saved quantum PCA to: {output_path}")
    print(f"U shape: {U.shape}")
    print(f"Explained variance (top-{args.k}): {np.sum(evr)*100:.2f}%")
    print(f"Per component: {evr}")

