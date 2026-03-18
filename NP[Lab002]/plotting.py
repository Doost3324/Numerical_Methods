import numpy as np
import matplotlib.pyplot as plt


def plot_newton_interpolation(x_data, y_data, coef, newton_func, x_pred=None, y_pred=None):
    x_min, x_max = min(x_data), max(x_data)
    x_dense = np.linspace(x_min, x_max, 500)
    y_dense = [newton_func(x_data, coef, xi) for xi in x_dense]

    plt.figure()

    # Orig data
    plt.scatter(x_data, y_data, label="Data points")

    # Interpolation
    plt.plot(x_dense, y_dense, label="Newton interpolation")

    # Prediction
    if x_pred is not None and y_pred is not None:
        plt.scatter([x_pred], [y_pred], label=f"Prediction ({x_pred}, {y_pred:.2f})")

    plt.xlabel("RPS")
    plt.ylabel("CPU")
    plt.title("CPU = f(RPS) via Newton Interpolation")
    plt.legend()
    plt.grid()

    plt.show()

def plot_omega(x_nodes, omega_func):

    x_min, x_max = min(x_nodes), max(x_nodes)
    x_dense = np.linspace(x_min, x_max, 500)
    y_dense = [omega_func(x_nodes, xi) for xi in x_dense]

    plt.figure()

    plt.plot(x_dense, y_dense, label="ω(x)")
    plt.axhline(0)  
    plt.scatter(x_nodes, [0]*len(x_nodes), label="Nodes")

    plt.title("Nodal Polynomial ω(x)")
    plt.xlabel("x")
    plt.ylabel("ω(x)")
    plt.legend()
    plt.grid()

    plt.show()

def plot_error(x_data, coef, x_dense_data, coef_dense, newton_func, error_func):
    import numpy as np
    import matplotlib.pyplot as plt

    x_min, x_max = min(x_data), max(x_data)
    x_vals = np.linspace(x_min, x_max, 500)

    errors = [
        error_func(xi, x_data, coef, x_dense_data, coef_dense, newton_func)
        for xi in x_vals
    ]

    plt.figure()

    plt.plot(x_vals, errors, label="Approximation Error")

    plt.title("Interpolation Error ε(x)")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()

    plt.show()

def plot_error_comparison(models, ref_model, newton_func, num_points=500):
    import numpy as np
    import matplotlib.pyplot as plt

    x_ref, coef_ref = ref_model
    x_min, x_max = min(x_ref), max(x_ref)
    x_vals = np.linspace(x_min, x_max, num_points)

    plt.figure()

    for label, (x_nodes, coef) in models.items():
        errors = []
        for xi in x_vals:
            try:
                err = abs(newton_func(x_ref, coef_ref, xi) - newton_func(x_nodes, coef, xi))
            except Exception:
                err = 0  
            errors.append(err)
        plt.plot(x_vals, errors, label=label)

    # plt.yscale("log")
    plt.title("Interpolation Error Comparison")
    plt.xlabel("RPS")
    plt.ylabel("Error (log scale)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_interpolation_comparison(models, original_data, newton_func):
    import numpy as np
    import matplotlib.pyplot as plt

    x_data, y_data = original_data

    x_min, x_max = min(x_data), max(x_data)
    x_vals = np.linspace(x_min, x_max, 500)

    plt.figure()

    # orig data
    plt.scatter(x_data, y_data, label="Original data")

    # plot interpolation
    for label, (x_nodes, coef) in models.items():
        y_vals = [newton_func(x_nodes, coef, xi) for xi in x_vals]
        plt.plot(x_vals, y_vals, label=label)

    plt.title("Interpolation Comparison (5, 10, 20 nodes)")
    plt.xlabel("RPS")
    plt.ylabel("CPU")
    plt.legend()
    plt.grid()

    plt.show()

def plot_factorial_interpolation(x_data, y_data, forward_func):
    import numpy as np
    import matplotlib.pyplot as plt

    x_min, x_max = min(x_data), max(x_data)
    x_vals = np.linspace(x_min, x_max, 500)

    y_vals = [forward_func(x_data, y_data, xi) for xi in x_vals]

    plt.figure()
    plt.scatter(x_data, y_data, label="Data points")
    plt.plot(x_vals, y_vals, label="Factorial (Newton Forward)")

    plt.title("Factorial Interpolation")
    plt.xlabel("RPS")
    plt.ylabel("CPU")
    plt.legend()
    plt.grid()
    plt.show()

def plot_newton_vs_factorial(x_data, y_data, coef, newton_func, forward_func):
    import numpy as np
    import matplotlib.pyplot as plt

    x_min, x_max = min(x_data), max(x_data)
    x_vals = np.linspace(x_min, x_max, 500)

    y_newton = [newton_func(x_data, coef, xi) for xi in x_vals]
    y_factorial = [forward_func(x_data, y_data, xi) for xi in x_vals]

    plt.figure()

    plt.scatter(x_data, y_data, label="Data points")
    plt.plot(x_vals, y_newton, label="Newton interpolation")
    plt.plot(x_vals, y_factorial, linestyle="--", label="Factorial interpolation")

    plt.title("Newton vs Factorial Interpolation")
    plt.xlabel("RPS")
    plt.ylabel("CPU")
    plt.legend()
    plt.grid()

    plt.show()

def plot_error_newton_vs_factorial(x_uniform, y_uniform,
                                  coef_newton,
                                  ref_model,
                                  newton_func,
                                  forward_func):
    import numpy as np
    import matplotlib.pyplot as plt

    x_ref, coef_ref = ref_model

    x_vals = np.linspace(min(x_ref), max(x_ref), 500)

    err_newton = []
    err_factorial = []

    for xi in x_vals:
        y_ref = newton_func(x_ref, coef_ref, xi)

        y_n = newton_func(x_uniform, coef_newton, xi)
        y_f = forward_func(x_uniform, y_uniform, xi)

        err_newton.append(abs(y_ref - y_n))
        err_factorial.append(abs(y_ref - y_f))

    plt.figure()

    plt.plot(x_vals, err_newton, label="Newton Error")
    plt.plot(x_vals, err_factorial, label="Factorial Error")

    plt.yscale("log")

    plt.title("Error Comparison: Newton vs Factorial")
    plt.xlabel("RPS")
    plt.ylabel("Error (log scale)")
    plt.legend()
    plt.grid()

    plt.show()

def plot_factorial_node_comparison(x_base, coef_base, newton_func, forward_func):
    import numpy as np
    import matplotlib.pyplot as plt

    x_5 = np.linspace(min(x_base), max(x_base), 5)
    y_5 = [newton_func(x_base, coef_base, xi) for xi in x_5]

    x_10 = np.linspace(min(x_base), max(x_base), 10)
    y_10 = [newton_func(x_base, coef_base, xi) for xi in x_10]

    x_20 = np.linspace(min(x_base), max(x_base), 20)
    y_20 = [newton_func(x_base, coef_base, xi) for xi in x_20]

    x_vals = np.linspace(min(x_base), max(x_base), 500)

    plt.figure()

    plt.scatter(x_5, y_5, label="Nodes (5)")

    plt.plot(x_vals, [forward_func(x_5, y_5, xi) for xi in x_vals], label="n=5")
    plt.plot(x_vals, [forward_func(x_10, y_10, xi) for xi in x_vals], label="n=10")
    plt.plot(x_vals, [forward_func(x_20, y_20, xi) for xi in x_vals], label="n=20")

    plt.title("Factorial Interpolation Comparison (5, 10, 20 nodes)")
    plt.xlabel("RPS")
    plt.ylabel("CPU")
    plt.legend()
    plt.grid()

    plt.show()