import matplotlib.pyplot as plt
import numpy as np

h_values = np.logspace(-8, -1, 20) 

def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def central_diff(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2*h)

def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

x0 = 10

R1_list = []
R2_list = []
R3_list = []

for h in h_values:
    exact = dM_exact(x0)

    # y'(h)
    y_h = central_diff(M, x0, h)

    # y'(2h)
    y_2h = central_diff(M, x0, 2*h)

    # R1 and R2
    R1 = abs(y_h - exact)
    R2 = abs(y_2h - exact)

    # Richardson (Runge)
    y_e = (4*y_h - y_2h) / 3
    R3 = abs(y_e - exact)

    R1_list.append(R1)
    R2_list.append(R2)
    R3_list.append(R3)