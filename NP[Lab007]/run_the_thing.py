import numpy as np
from delusion import jacobi, gauss_seidel, simple_iteration

A = np.loadtxt("A.txt")
b = np.loadtxt("B.txt")

n = len(b)
x0 = np.ones(n)

x_j, it_j, eps_j = jacobi(A, b, x0)
x_gs, it_gs, eps_gs = gauss_seidel(A, b, x0)
x_s, it_s, eps_s = simple_iteration(A, b, x0)

# np.savetxt("x_jacobi.txt", x_j)
# np.savetxt("x_seidel.txt", x_gs) # <--- +10% load time
# np.savetxt("x_simple.txt", x_s)

print("\nRes:")
print("Jacobi: iterations =", it_j, "eps =", eps_j)
print("Gauss-Seidel: iterations =", it_gs, "eps =", eps_gs)
print("Simple Iteration: iterations =", it_s, "eps =", eps_s)