EPS = 1e-10
MAX_ITER = 1000
import cmath

def read_coeffs(filename):
    with open(filename, "r") as f:
        coeffs = list(map(float, f.read().split()))
    return coeffs 

def horner(coeffs, x):
    result = coeffs[0]
    for c in coeffs[1:]:
        result = result * x + c
    return result

def derivative_coeffs(coeffs):
    n = len(coeffs) - 1
    return [coeffs[i] * (n - i) for i in range(n)]

def newton_poly(coeffs, x0):
    dcoeffs = derivative_coeffs(coeffs)

    x = x0
    iterations = 0

    while iterations < MAX_ITER:
        fx = horner(coeffs, x)
        dfx = horner(dcoeffs, x)

        if dfx == 0:
            break

        x_new = x - fx / dfx

        if abs(fx) < EPS and abs(x_new - x) < EPS:
            return x_new, iterations

        x = x_new
        iterations += 1

    return x, iterations

#Complex roots

def lin_deflate(coeffs, root):
    new_coeffs = [coeffs[0]]

    for i in range(1, len(coeffs) - 1):
        new_coeffs.append(new_coeffs[-1] * root + coeffs[i])

    return new_coeffs 

def solve_quadratic(a, b, c):
    D = b**2 - 4*a*c
    x1 = (-b + cmath.sqrt(D)) / (2*a)
    x2 = (-b - cmath.sqrt(D)) / (2*a)
    return x1, x2

coeffs = read_coeffs("coeffs.txt")

real_root, iters = newton_poly(coeffs, 0)

print("Real root(Horner):", real_root)
print("Iterations:", iters)

quad = lin_deflate(coeffs, real_root)
x1, x2 = solve_quadratic(quad[0], quad[1], quad[2])

print("Complex roots(Lin):", x1, x2)