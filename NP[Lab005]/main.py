import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy.special import erf

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

x = np.linspace(0, 24, 1000)
y = f(x)

def exact_integral():
    term1 = 1200
    term2 = 0
    term3 = 5 * sqrt(pi / 0.2) * erf(12 * sqrt(0.2))
    return term1 + term2 + term3

I0 = exact_integral()
print("Exact integral I0 =", I0)

def simpson_integral(f, a, b, N):
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's rule")

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)

    I = (h / 3) * (
        y[0]
        + 4 * np.sum(y[1:-1:2])
        + 2 * np.sum(y[2:-1:2])
        + y[-1]
    )

    return I

a, b = 0, 24
N_values = list(range(10, 1001, 2)) 

errors = [
    abs(simpson_integral(f, a, b, N) - I0)
    for N in N_values
]

errors_RR = []
for N in N_values:
    if N % 4 != 0:  
        errors_RR.append(np.nan)
        continue

    I_N = simpson_integral(f, a, b, N)
    I_half = simpson_integral(f, a, b, N // 2)

    I_RR = I_N + (I_N - I_half) / 15
    errors_RR.append(max(abs(I_RR - I0), 1e-16))

eps_target = 1e-12
N_opt = None

for N, err in zip(N_values, errors):
    if err < eps_target:
        N_opt = N
        break

N0 = 16

I_N0 = simpson_integral(f, a, b, N0)
eps0 = abs(I_N0 - I0)

N_half = N0 // 2  # 8
I_half = simpson_integral(f, a, b, N_half)

I_coarse = simpson_integral(f, a, b, N_half)
I_fine   = simpson_integral(f, a, b, N0)

I_R = (16 * I_fine - I_coarse) / 15
eps_R = abs(I_R - I0)

print("N_opt =", N_opt)
print("Achieved error =", errors[N_values.index(N_opt)])

print("N0 =", N0)
print("I(N0) =", I_N0)
print("eps0 =", eps0)

print("I(N0) =", I_N0)
print("I(N0/2) =", I_half)
print("Runge-Romberg I_R =", I_R)
print("eps_R =", eps_R)

# print(len(N_values), len(errors), len(errors_RR))

# print("RR valid count:", np.sum(~np.isnan(errors_RR)))
# print("RR min:", np.nanmin(errors_RR))
# print("RR max:", np.nanmax(errors_RR))

N1 = N0
N2 = N1 * 2
N3 = N2 * 2

I1 = simpson_integral(f, a, b, N1)
I2 = simpson_integral(f, a, b, N2)
I3 = simpson_integral(f, a, b, N3)

p = np.log2(abs((I1 - I2) / (I2 - I3)))

I_E = I3 + (I3 - I2) / (2**p - 1)

eps_E = abs(I_E - I0)

Ait_I_check = I2 - I3

print("Aitken p =", p)
print("I_E =", I_E)
print("eps_E =", eps_E)
# print("I2 - I3 check", Ait_I_check)

# print(N1, N2, N3)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(N_values, errors)
plt.ylim(1e-16, 1e1)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('ε(N) = |I(N) - I0|')
plt.title('Залежність похибки від кількості розбиттів (метод Сімпсона)')
plt.grid(True)
plt.show()

errors_RR = np.array(errors_RR, dtype=float)
errors_RR[errors_RR == 0] = np.nan

plt.figure(figsize=(10, 6))

plt.plot(N_values, errors, label='Simpson')

mask = ~np.isnan(errors_RR)
plt.plot(np.array(N_values)[mask], errors_RR[mask], label='Runge-Romberg')

plt.ylim(1e-16, 1e1)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Порівняння точності методів')
plt.legend()
plt.grid(True)

plt.show()

N_values = np.array(N_values)

h = (b - a) / N_values

ref = h**4
ref = ref * (errors[0] / ref[0]) 

errors = np.array([
    abs(simpson_integral(f, a, b, N) - I0)
    for N in N_values
])

errors_RR = []

for N in N_values:
    if N % 4 != 0:
        errors_RR.append(np.nan)
        continue

    I_N = simpson_integral(f, a, b, N)
    I_half = simpson_integral(f, a, b, N // 2)

    I_RR = I_N + (I_N - I_half) / 15
    errors_RR.append(abs(I_RR - I0))

errors_RR = np.array(errors_RR, dtype=float)

#Aitken v1

errors_Aitken = []

N_set = set(N_values)

for N in N_values:
    if (2*N not in N_set) or (4*N not in N_set):
        errors_Aitken.append(np.nan)
        continue

    I1 = simpson_integral(f, a, b, N)
    I2 = simpson_integral(f, a, b, 2*N)
    I3 = simpson_integral(f, a, b, 4*N)

    p = np.log2(abs((I1 - I2) / (I2 - I3)))

    I_E = I3 + (I3 - I2) / (2**p - 1)

    errors_Aitken.append(abs(I_E - I0))

errors_Aitken = np.array(errors_Aitken, dtype=float)

plt.figure(figsize=(10, 6))

mask_simpson = ~np.isnan(errors)
mask_rr = ~np.isnan(errors_RR)
mask_aitken = ~np.isnan(errors_Aitken)

plt.loglog(h[mask_simpson], errors[mask_simpson], 'o-', label='Simpson')
plt.loglog(h[mask_rr], errors_RR[mask_rr], 's-', label='Runge–Romberg')
plt.loglog(h[mask_aitken], errors_Aitken[mask_aitken], '^-', label='Aitken')

plt.loglog(h, ref, '--', label='O(h^4)')

# plt.gca().invert_xaxis()

plt.xlabel('h (step size)')
plt.ylabel('Error')
plt.title('Порівняння похибки методів')
plt.legend()
plt.grid(True, which='both')

plt.show()