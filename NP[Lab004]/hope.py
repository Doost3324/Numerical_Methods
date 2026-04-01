import matplotlib.pyplot as plt
import numpy as np
from plot import *

#P1
y_val = []
x_val = []

def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

for t in range(20):
    y_val.append(M(t))
    x_val.append(t)

plt.plot(x_val, y_val)
plt.ylabel("M(t)")
plt.xlabel("t")
plt.grid("both")
plt.show()

def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

t = x_val[0]
print(dM_exact(x_val[0]))

h_grid = x_val[1] - x_val[0]

y_val = np.array(y_val)
x_val = np.array(x_val)

d_forward = []
d_central = []

for i in range(len(x_val) - 1):
    h_grid = x_val[1] - x_val[0] 
    d_forward.append(h_grid)

for i in range(1, len(x_val) - 1):
    val = (y_val[i+1] - y_val[i-1]) / (2*h_grid)
    d_central.append(val)

d_exact = dM_exact(x_val)

x_forward = x_val[:-1]
x_central = x_val[1:-1]

d_exact_forward = d_exact[:-1]
d_exact_central = d_exact[1:-1]

error_forward = np.abs(d_forward - d_exact_forward)
error_central = np.abs(d_central - d_exact_central)

x0_index = 10
x0 = x_val[x0_index]

y_prime_h = (M(x0 + h) - M(x0 - h)) / (2 * h)

exact = dM_exact(x0)
R1 = abs(y_prime_h - exact)

print("R1 =", R1)

y_prime_2h = (M(x0 + 2*h) - M(x0 - 2*h)) / (4 * h)

R2 = abs(y_prime_2h - exact)
print("R2 =", R2)

y_e = (4 * y_prime_h - y_prime_2h) / 3

R3 = abs(y_e - exact)

print("R3 =", R3)

y_h  = (M(x0 + h)   - M(x0 - h))   / (2*h)
y_2h = (M(x0 + 2*h) - M(x0 - 2*h)) / (4*h)
y_4h = (M(x0 + 4*h) - M(x0 - 4*h)) / (8*h)

y_E = (y_2h**2 - y_4h * y_h) / (2*y_2h - (y_4h + y_h))

p = (1 / np.log(2)) * np.log(abs((y_4h - y_2h) / (y_2h - y_h)))

print("Order p =", p)

R_final = abs(y_E - exact)
print("Final error =", R_final)

min_index = np.argmin(R1_list)
print("Optimal h ≈", h_values[min_index])

plt.figure(figsize=(8,5))
plt.loglog(h_values, R1_list, marker='o')
plt.xlabel("h")
plt.ylabel("R1")
plt.title("R1(h)")
plt.show()

plt.figure(figsize=(8,5))
plt.loglog(h_values, R2_list, marker='o')
plt.xlabel("h")
plt.ylabel("R2")
plt.title("R2(h)")
plt.show()

plt.figure(figsize=(8,5))
plt.loglog(h_values, R3_list, marker='o')
plt.xlabel("h")
plt.ylabel("R3")
plt.title("R3(h) (after Richardson)")
plt.show()

plt.figure(figsize=(10,6))

plt.loglog(h_values, R1_list, marker='o', label="R1 (h)")
plt.loglog(h_values, R2_list, marker='s', label="R2 (2h)")
plt.loglog(h_values, R3_list, marker='^', label="R3 (Richardson)")

plt.xlabel("h")
plt.ylabel("Error")
plt.title("Error vs Step Size")
plt.legend()

plt.show()