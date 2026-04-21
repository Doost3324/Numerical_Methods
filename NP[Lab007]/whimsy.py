import numpy as np

N = 100

def generate_matrix(n):
    A = np.random.rand(n, n)

    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + 1  

    return A

def generate_exact_solution(n, value=200.4):
    return np.full(n, value)

def compute_b(A, x):
    return A @ x

def save_matrix(filename, A):
    np.savetxt(filename, A)

def save_vector(filename, v):
    np.savetxt(filename, v)

A = generate_matrix(N)
x_true = generate_exact_solution(N)
b = compute_b(A, x_true)

save_matrix("A.txt", A)
save_vector("B.txt", b)

