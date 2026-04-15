import numpy as np

def load_matrix(filename):
    return np.loadtxt(filename)

def load_vector(filename):
    return np.loadtxt(filename)

def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Upper triangular
        for j in range(i, n):
            sum_u = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum_u

        # Lower triangular
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                sum_l = sum(L[j][k] * U[k][i] for k in range(i))
                L[j][i] = (A[j][i] - sum_l) / U[i][i]

    return L, U

# ---------- Solve Ly = b ----------
def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y

# ---------- Solve Ux = y ----------
def back_substitution(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def matvec(A, x):
    return A @ x

def vector_norm(v):
    return np.linalg.norm(v) 

def solve_lu(A, b):
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x, L, U

def save_lu(L, U):
    np.savetxt("L.txt", L)
    np.savetxt("U.txt", U)

A = load_matrix("A.txt")
b = load_vector("B.txt")

x_lu, L, U = solve_lu(A, b)

save_lu(L, U)
np.savetxt("x_lu.txt", x_lu)

# x_true = np.full(100, 2.5)
# error = np.linalg.norm(x_lu - x_true)
# print("Error vs true solution:", error)