import numpy as np
import sys
import os

# Add project directories to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quantum_src'))

# Import our safe and unsafe functions
from classifier import grover_search_minimum, grover_search_minimum_raw_unsafe

# Create a sample list of distances. Let's make it 50 items long.
# We'll hide the minimum value at a random index.
sample_distances = np.random.rand(50) * 100
min_index_to_find = np.random.randint(0, 50)
sample_distances[min_index_to_find] = 1.0 # The minimum value

# 1. Find the answer classically
classical_result = int(np.argmin(sample_distances))
print(f"The correct answer (from classical argmin) is: {classical_result}\n")

# 2. Run the "safe" quantum function with the fallback
safe_quantum_result = grover_search_minimum(sample_distances)
print("--- Running the SAFE quantum function (with fallback) ---")
print(f"Result: {safe_quantum_result}")
if safe_quantum_result == classical_result:
    print("✅ As expected, the safe function returned the correct answer.\n")
else:
    print("❌ Something is wrong with the safe function.\n")


# 3. Run the "raw" quantum function WITHOUT the fallback
raw_quantum_result = grover_search_minimum_raw_unsafe(sample_distances)
print("--- Running the RAW quantum function (NO fallback) ---")
print(f"Result: {raw_quantum_result}")
if raw_quantum_result == classical_result:
    print("✅ PROOF! The quantum circuit successfully found the correct minimum value on its own!\n")
else:
    print(f"⚠️  The quantum circuit was noisy and found {raw_quantum_result} instead of {classical_result}. The fallback in the safe version would have corrected this.\n")

