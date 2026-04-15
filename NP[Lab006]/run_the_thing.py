from iteration import gauss_seidel, compute_eps
import numpy as np

# Load data
A = np.loadtxt("matrix.txt")
b = np.loadtxt("mat_size.txt")
x_lu = np.loadtxt("x_lu.txt")

# Initial approximation (IMPORTANT: assignment requirement)
x0 = x_lu.copy()

# Run iterative refinement
x_iter, iterations, final_eps = gauss_seidel(A, b, x0)

# Save result
np.savetxt("x_iter.txt", x_iter)

print("\n--- Iterative Method Results ---")
print("Iterations:", iterations)
print("Final eps:", final_eps)

# Compare with true solution
x_true = np.full(100, 2.5)
error = np.linalg.norm(x_iter - x_true)
print("Error vs true:", error)