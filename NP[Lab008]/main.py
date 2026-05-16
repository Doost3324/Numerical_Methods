import numpy as np
np.set_printoptions(legacy='1.25')

def derivative_sign(x):
    h = 1e-5
    return (F(x + h) - F(x)) / h

def F(x):
    return np.cos(x) - x**2 + 0.5

a = -2
b = 2
h = 0.1

x_values = np.arange(a, b + h, h)
y_values = F(x_values)

with open("tabulation.txt", "w") as f:
    for x, y in zip(x_values, y_values):
        f.write(f"{x:.3f} {y:.6f}\n")

roots_intervals = []

for i in range(len(x_values) - 1):
    if y_values[i] * y_values[i + 1] < 0:
        roots_intervals.append((x_values[i], x_values[i + 1]))

print("Approximate root intervals:")
for interval in roots_intervals:
    print(interval)

intervals = roots_intervals
