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
# np.savetxt("x_seidel.txt", x_gs)
# np.savetxt("x_simple.txt", x_s)

print("\nRes:")
print("Jacobi: iterations =", it_j, "eps =", eps_j,  "x:", np.mean(x_j))
print("Gauss-Seidel: iterations =", it_gs, "eps =", eps_gs, "x:", np.mean(x_gs))
print("Simple Iteration: iterations =", it_s, "eps =", eps_s, "x:", np.mean(x_s))