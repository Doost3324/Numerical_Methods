import csv
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
    return x, y


def normalize_x(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std, mean, std

def form_matrix(x, m):
    n = len(x)
    A = np.zeros((m+1, m+1))

    for i in range(m+1):
        for j in range(m+1):
            A[i, j] = sum(x[k]**(i+j) for k in range(n))

    return A

def form_vector(x, y, m):
    n = len(x)
    b = np.zeros(m+1)

    for i in range(m+1):
        b[i] = sum(y[k] * x[k]**i for k in range(n))

    return b


def gauss_solve(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    for k in range(n):
        max_row = max(range(k, n), key=lambda i: abs(A[i, k]))

        if abs(A[max_row, k]) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned")

        A[[k, max_row]] = A[[max_row, k]]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x_sol = np.zeros(n)
    for i in range(n-1, -1, -1):
        x_sol[i] = (b[i] - sum(A[i, j]*x_sol[j] for j in range(i+1, n))) / A[i, i]

    return x_sol

def polynomial(x, coef):
    y = np.zeros_like(x, dtype=float)
    for i in range(len(coef)):
        y += coef[i] * x**i
    return y

def variance(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


x, y = read_data("data.csv")
print(x, y)
x_norm, mean_x, std_x = normalize_x(x)
print(x_norm)
variances = []
coefficients = []

max_degree = 4

for m in range(1, max_degree + 1):
    A = form_matrix(x_norm, m)
    b = form_vector(x_norm, y, m)
    coef = gauss_solve(A, b)

    y_approx = polynomial(x_norm, coef)
    var = variance(y, y_approx)

    variances.append(var)
    coefficients.append(coef)

print(variances)

optimal_m = np.argmin(variances) + 1
optimal_coef = coefficients[optimal_m - 1]

n = len(x)
h1 = (x[-1] - x[0]) / (20 * n)

x_dense = np.arange(x[0], x[-1], h1)

x_dense_norm = (x_dense - mean_x) / std_x

y_dense = polynomial(x_dense_norm, optimal_coef)

y_true_dense = np.interp(x_dense, x, y)

error_dense = y_true_dense - y_dense

print("Variances:")
for i, v in enumerate(variances, start=1):
    print(f"m={i}: {v:.4f}")

print("\nOptimal degree:", optimal_m)

y_approx = polynomial(x_norm, optimal_coef)

error = y - y_approx

x_future = np.array([25, 26, 27])
x_future_norm = (x_future - mean_x) / std_x
y_future = polynomial(x_future_norm, optimal_coef)


print("\nForecast for next 3 months:")
for xi, yi in zip(x_future, y_future):
    print(f"Month {int(xi)}: {yi:.2f}")

plt.figure()
plt.scatter(x, y, label="Actual data")
plt.plot(x, y_approx, label=f"Polynomial m={optimal_m}")

plt.scatter(x_future, y_future, color='red', label="Predicted (forecast)", zorder=5)

plt.plot([x[-1], *x_future], [y_approx[-1], *y_future], color='red', linestyle='--', alpha=0.7)

plt.legend()
plt.title("Temperature Approximation & Forecast")
plt.xlabel("Month")
plt.ylabel("Temperature")
plt.grid()

plt.figure()
plt.plot(range(1, max_degree+1), variances, marker='o')
plt.title("Variance vs Polynomial Degree")
plt.xlabel("Degree m")
plt.ylabel("Variance")
plt.grid()

plt.figure()
plt.plot(x, error, marker='o')
plt.title("Approximation Error")
plt.xlabel("Month")
plt.ylabel("Error")
plt.grid()

plt.figure()
plt.plot(x_dense, error_dense)
plt.title("Tabulated Approximation Error")
plt.xlabel("Month")
plt.ylabel("Error")
plt.grid()

plt.show()
