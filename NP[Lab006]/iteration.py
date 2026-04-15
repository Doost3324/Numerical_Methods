import numpy as np

def compute_eps(A, x, b):
    r = A @ x - b
    return np.linalg.norm(r, np.inf) / np.linalg.norm(b, np.inf)

def gauss_seidel(A, b, x0, eps_target=1e-14, max_iter=2000, stagnation_tol=1e-16):
    n = len(b)
    x = x0.copy()

    eps = compute_eps(A, x, b)
    print(f"Initial eps = {eps}")

    if eps < eps_target:
        return x, 0, eps

    prev_eps = eps

    for k in range(max_iter):
        x_old = x.copy()
        x_new = x.copy()

        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])    
            s2 = np.dot(A[i, i+1:], x[i+1:])     
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        x = x_new

        eps = compute_eps(A, x, b)

        if k % 10 == 0:
            print(f"Iter {k}: eps = {eps}")

        if abs(prev_eps - eps) < stagnation_tol:
            print("Stagnation detected")
            return x, k + 1, eps

        prev_eps = eps

        if eps < eps_target:
            return x, k + 1, eps

    return x, max_iter, eps

A = np.loadtxt("A.txt")
b = np.loadtxt("B.txt")
x_lu = np.loadtxt("x_lu.txt")

x0 = np.zeros_like(b)

x_iter, iterations, final_eps = gauss_seidel(A, b, x0)

np.savetxt("x_iter.txt", x_iter)

print("\nRes:")
print("Iterations:", iterations)
print("Final eps:", final_eps)

# x_true = np.full(100, 200.4)
# error = np.linalg.norm(x_iter - x_true)
# print("Error vs true:", error) 

# print("Check residual of LU solution:", np.max(np.abs(A @ x_lu - b)))
