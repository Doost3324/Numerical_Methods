import csv
from implementation import *
from plotting import *

def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['RPS']))
            y.append(float(row['CPU']))
    return x, y

x, y = read_data("data.csv")
print("x:", x)
print("y:", y)

table_5 = divided_differences(x, y)
coef_5 = get_newton_coefficients(table_5)

print("\nDivided Differences Table (5 nodes):")
print(table_5)
print("\nNewton Coefficients (5 nodes):")
print(coef_5)

x_pred = 600
y_pred = newton_polynomial(x, coef_5, x_pred)
print(f"Prediction at 600 RPS: {y_pred}")

plot_newton_interpolation(
    x_data=x,
    y_data=y,
    coef=coef_5,
    newton_func=newton_polynomial,
    x_pred=x_pred,
    y_pred=y_pred
)

# for xi in x:
#     print(f"omega({xi}) =", omega(x, xi))
# plot_omega(x, omega)

x_dense, y_dense = generate_dense_nodes(x, coef_5, newton_polynomial, 20)
table_dense = divided_differences(x_dense, y_dense)
coef_dense = get_newton_coefficients(table_dense)

plot_error(
    x_data=x,
    coef=coef_5,
    x_dense_data=x_dense,
    coef_dense=coef_dense,
    newton_func=newton_polynomial,
    error_func=interpolation_error
)

table_5 = divided_differences(x, y)
coef_5 = get_newton_coefficients(table_5)

x_ref, y_ref = generate_dense_nodes(x, coef_5, newton_polynomial, 50)
table_ref = divided_differences(x_ref, y_ref)
coef_ref = get_newton_coefficients(table_ref)

x_10, y_10 = generate_dense_nodes(x, coef_5, newton_polynomial, 10)
table_10 = divided_differences(x_10, y_10)
coef_10 = get_newton_coefficients(table_10)

x_20, y_20 = generate_dense_nodes(x, coef_5, newton_polynomial, 20)
table_20 = divided_differences(x_20, y_20)
coef_20 = get_newton_coefficients(table_20)

plot_error_comparison(
    models={
        "n=5": (x, coef_5),
        "n=10": (x_10, coef_10),
        "n=20": (x_20, coef_20),
    },
    ref_model=(x_ref, coef_ref),
    newton_func=newton_polynomial
)

plot_interpolation_comparison(
    models={
        "n=5": (x, coef_5),
        "n=10": (x_10, coef_10),
        "n=20": (x_20, coef_20),
    },
    original_data=(x, y),
    newton_func=newton_polynomial
)

x_uniform, y_uniform = make_uniform_grid(x, coef_5, newton_polynomial, len(x))

plot_factorial_interpolation(
    x_data=x_uniform,
    y_data=y_uniform,
    forward_func=newton_forward
)

x_uniform, y_uniform = make_uniform_grid(x, coef_5, newton_polynomial, len(x))

plot_newton_vs_factorial(
    x_uniform, y_uniform,
    coef_5,
    newton_polynomial,
    newton_forward
)

plot_error_newton_vs_factorial(
    x_uniform, y_uniform,
    coef_5,
    (x_ref, coef_ref),
    newton_polynomial,
    newton_forward
)

plot_factorial_node_comparison(
    x, coef_5,
    newton_polynomial,
    newton_forward
)