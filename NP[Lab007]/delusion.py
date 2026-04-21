import numpy as np

max_iter = 10000

def norm(v):
    return np.linalg.norm(v, ord=np.inf)

def compute_eps(A, x, b):
    r = A @ x - b
    return norm(r) / norm(b)


def jacobi(A, b, x0, eps=1e-14):
    n = len(b)
    x = x0.copy()
    x_new = np.zeros_like(x)

    for k in range(max_iter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        if compute_eps(A, x_new, b) < eps:
            return x_new, k + 1, compute_eps(A, x_new, b)

        x = x_new.copy()

    return x, max_iter, compute_eps(A, x, b)


def gauss_seidel(A, b, x0, eps=1e-14):
    n = len(b)
    x = x0.copy()

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i+1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]

        if compute_eps(A, x, b) < eps:
            return x, k + 1, compute_eps(A, x, b)

    return x, max_iter, compute_eps(A, x, b)


def simple_iteration(A, b, x0, eps=1e-14):
    x = x0.copy()
    tau = 1.0 / np.max(np.sum(np.abs(A), axis=1))

    for k in range(max_iter):
        x_new = x + tau * (b - A @ x)

        if compute_eps(A, x_new, b) < eps:
            return x_new, k + 1, compute_eps(A, x_new, b)

        x = x_new

    return x, max_iter, compute_eps(A, x, b)
