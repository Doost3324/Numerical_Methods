import numpy as np

def divided_differences(x: list[float], y: list[float]) -> np.ndarray:
    n = len(x)
    table = np.zeros((n, n))

    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            numerator = table[i + 1][j - 1] - table[i][j - 1]
            denominator = x[i + j] - x[i]
            table[i][j] = numerator / denominator

    return table


def get_newton_coefficients(table: np.ndarray) -> np.ndarray:

    return table[0]

def newton_polynomial(x_data: list[float], coef: list[float], x: float) -> float:

    n = len(coef)
    result = coef[0]
    term = 1.0

    for i in range(1, n):
        term *= (x - x_data[i - 1])
        result += coef[i] * term

    return result

def omega(x_nodes: list[float], x: float) -> float:

    result = 1.0
    for xi in x_nodes:
        result *= (x - xi)
    return result

def generate_dense_nodes(x_data, coef, newton_func, num_points=20):

    import numpy as np

    x_min, x_max = min(x_data), max(x_data)
    x_dense = np.linspace(x_min, x_max, num_points)
    y_dense = [newton_func(x_data, coef, xi) for xi in x_dense]

    return list(x_dense), list(y_dense)

def generate_nodes_and_coefficients(x_data, y_data, num_nodes):
    import numpy as np
    indices = np.linspace(0, len(x_data) - 1, num_nodes, dtype=int)
    x_nodes = [x_data[i] for i in indices]
    y_nodes = [y_data[i] for i in indices]
    table = divided_differences(x_nodes, y_nodes)
    coef = get_newton_coefficients(table)
    return x_nodes, coef

def interpolation_error(x, x_data_1, coef_1, x_data_2, coef_2, newton_func):

    y1 = newton_func(x_data_1, coef_1, x)
    y2 = newton_func(x_data_2, coef_2, x)
    return abs(y2 - y1)

def finite_differences(y):
    n = len(y)
    table = [y.copy()]

    for i in range(1, n):
        row = []
        for j in range(n - i):
            diff = table[i - 1][j + 1] - table[i - 1][j]
            row.append(diff)
        table.append(row)

    return table

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def newton_forward(x_data, y_data, x):
    h = x_data[1] - x_data[0]  
    t = (x - x_data[0]) / h

    diff_table = finite_differences(y_data)

    result = y_data[0]
    t_term = 1

    for i in range(1, len(x_data)):
        t_term *= (t - (i - 1))
        result += (t_term / factorial(i)) * diff_table[i][0]

    return result

def make_uniform_grid(x_data, coef, newton_func, n_points):

    x_uniform = np.linspace(min(x_data), max(x_data), n_points)
    y_uniform = [newton_func(x_data, coef, xi) for xi in x_uniform]

    return x_uniform, y_uniform