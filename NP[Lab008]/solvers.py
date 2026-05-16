import numpy as np
from main import intervals

EPS = 1e-10
MAX_ITER = 1000

def F(x):
    return np.cos(x) - x**2 + 0.5

def d2F(x):
    h = 1e-5
    return (F(x + h) - 2*F(x) + F(x - h)) / (h**2)


def stop_condition(x_new, x_old):
    return abs(F(x_new)) < EPS and abs(x_new - x_old) < EPS

def bisection(a, b):
    if F(a) * F(b) > 0:
        raise ValueError("No sign change[interval]")

    iterations = 0

    while iterations < MAX_ITER:
        c = (a + b) / 2

        if abs(F(c)) < EPS:
            return c, iterations

        if F(a) * F(c) < 0:
            b = c
        else:
            a = c

        iterations += 1

    return c, iterations

def dF(x):
    h = 1e-6
    return (F(x + h) - F(x - h)) / (2 * h)

def newton(x0):
    x = x0
    iterations = 0

    while iterations < MAX_ITER:
        x_new = x - F(x) / dF(x)

        if stop_condition(x_new, x):
            return x_new, iterations

        x = x_new
        iterations += 1

    return x, iterations

def secant_chord(x0, x1):
    iterations = 0

    while iterations < MAX_ITER:
        if F(x1) - F(x0) == 0:
            break

        x_new = x1 - F(x1) * (x1 - x0) / (F(x1) - F(x0))

        if stop_condition(x_new, x1):
            return x_new, iterations

        x0, x1 = x1, x_new
        iterations += 1

    return x1, iterations

def phi_positive(x):
    val = np.cos(x) + 0.5
    if val < 0:
        return None
    return np.sqrt(val)

def phi_negative(x):
    val = np.cos(x) + 0.5
    if val < 0:
        return None
    return -np.sqrt(val)

def simple_iteration(x0, phi_func):
    x = x0
    iterations = 0

    while iterations < MAX_ITER:
        x_new = phi_func(x)

        if x_new is None:
            raise ValueError("Invalid iteration")

        if stop_condition(x_new, x):
            return x_new, iterations

        x = x_new
        iterations += 1

    return x, iterations

def chebyshev(x0):
    x = x0
    iterations = 0

    while iterations < MAX_ITER:
        fx = F(x)
        dfx = dF(x)
        d2fx = d2F(x)

        if dfx == 0:
            break

        x_new = x - fx/dfx - (fx**2 * d2fx) / (2 * dfx**3)

        if stop_condition(x_new, x):
            return x_new, iterations

        x = x_new
        iterations += 1

    return x, iterations

def parabolic(x0, x1, x2):
    iterations = 0

    while iterations < MAX_ITER:
        f0, f1, f2 = F(x0), F(x1), F(x2)

        # coefficients of interpolating parabola
        denom = (x0 - x1)*(x0 - x2)*(x1 - x2)
        if denom == 0:
            break

        A = (x2*(f1 - f0) + x1*(f0 - f2) + x0*(f2 - f1)) / denom
        B = (x2**2*(f0 - f1) + x1**2*(f2 - f0) + x0**2*(f1 - f2)) / denom

        if A == 0:
            break

        x_new = -B / (2*A)

        if stop_condition(x_new, x2):
            return x_new, iterations

        x0, x1, x2 = x1, x2, x_new
        iterations += 1

    return x2, iterations

def inverse_interpolation(x0, x1, x2):
    iterations = 0

    while iterations < MAX_ITER:
        f0, f1, f2 = F(x0), F(x1), F(x2)

        if (f0 == f1) or (f1 == f2) or (f0 == f2):
            break

        x_new = (
            x0 * f1 * f2 / ((f0 - f1)*(f0 - f2)) +
            x1 * f0 * f2 / ((f1 - f0)*(f1 - f2)) +
            x2 * f0 * f1 / ((f2 - f0)*(f2 - f1))
        )

        if stop_condition(x_new, x2):
            return x_new, iterations

        x0, x1, x2 = x1, x2, x_new
        iterations += 1

    return x2, iterations

(a1, b1) = intervals[0]
(a2, b2) = intervals[1]
mid1 = (a1 + b1) / 2
mid2 = (a2+ b2) / 2

print("R1")
# print("Bisection:", bisection(a1, b1))
print("Newton:", newton((a1 + b1)/2))
print("Chord:", secant_chord(a1, b1))
print("Simple Iteration:", simple_iteration((a1 + b1)/2, phi_negative))

print("Chebyshev:", chebyshev(mid1))
print("Parabolic:", parabolic(a1, mid1, b1))
print("Inverse Interpolation:", inverse_interpolation(a1, mid1, b1))

print("\nR2")
# print("Bisection:", bisection(a2, b2))
print("Newton:", newton((a2 + b2)/2))
print("Chord:", secant_chord(a2, b2))
print("Simple Iteration:", simple_iteration((a2 + b2)/2, phi_positive))
print("Chebyshev:", chebyshev(mid2))
print("Parabolic:", parabolic(a2, mid2, b2))
print("Inverse Interpolation:", inverse_interpolation(a2, mid2, b2))