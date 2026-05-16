from main import *

# # 1. Define the Rosenbrock objective function
# def rosenbrock(x):
#     # Python uses 0-based indexing, so x[0] is x_1 and x[1] is x_2
#     term1 = 100.0 * (x[0]**2 - x[1])**2
#     term2 = (x[0] - 1.0)**2
#     return term1 + term2

# # 2. Set the initial parameters based on the assignment image
# x0_rosenbrock = [-1.2, 0.0]

# # 3. Set the starting step sizes (a standard starting value is 0.5 or 1.0)
# initial_steps = [0.5, 0.5]

# # 4. Run the Hooke-Jeeves optimization
# best_x, best_f, trajectory, steps = hooke_jeeves(
#     func=rosenbrock, 
#     x0=x0_rosenbrock, 
#     initial_steps=initial_steps,
#     step_reduce_factor=2.0, 
#     epsilon=1e-5
# )

# # 5. Output the results
# print("--- Hooke-Jeeves Optimization: Rosenbrock Function ---")
# print(f"Optimization finished in {steps} iterations.")
# print(f"Found Optimal Point (x1, x2): [{best_x[0]:.5f}, {best_x[1]:.5f}]")
# print(f"Minimum Function Value: {best_f:.15f}")

import numpy as np

# --- 1. Define all target functions from the image ---

def rosenbrock(x):
    return 100.0 * (x[0]**2 - x[1])**2 + (x[0] - 1.0)**2

def power_func(x):
    return (10.0 * (x[0] - x[1])**2 + (x[0] - 1.0)**2)**4

def root_func(x):
    return (10.0 * (x[0] - x[1])**2 + (x[0] - 1.0)**2)**0.25

def wood_func(x):
    return (100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2 + 
            90.0 * (x[3] - x[2]**2)**2 + (1.0 - x[2])**2 + 
            10.1 * ((x[1] - 1.0)**2 + (x[3] - 1.0)**2) + 
            19.8 * (x[1] - 1.0) * (x[3] - 1.0))

def powell_func(x):
    return ((x[0] + 10.0 * x[1])**2 + 5.0 * (x[2] - x[3])**2 + 
            10.0 * (x[0] - x[3])**4 + (x[1] - 2.0 * x[2])**4)

def miele_func(x):
    return ((np.exp(x[0]) - x[1])**4 + 100.0 * (x[1] - x[2])**6 + 
            (np.tan(x[2] - x[3]))**4 + x[0]**8 + (x[3] - 1.0)**2)


# --- 2. Setup the test suite ---

# List of tuples: (Function Name, Function Object, Initial Guess X0)
test_functions = [
    ("1. Rosenbrock Function", rosenbrock, [-1.2, 0.0]),
    ("2. Power Function", power_func, [-1.2, 0.0]),
    ("3. Root Function", root_func, [-1.2, 0.0]),
    ("4. Wood Function", wood_func, [-3.0, -1.0, -3.0, -1.0]),
    ("5. Powell Function", powell_func, [-3.0, -1.0, 0.0, 1.0]),
    ("6. Miele Function", miele_func, [1.0, 2.0, 2.0, 2.0])
]

print("=== Testing All Target Functions with Hooke-Jeeves ===\n")

# --- 3. Loop through and solve ---

for name, func, x0 in test_functions:
    dimensions = len(x0)
    
    # Dynamically create the initial step sizes based on the number of dimensions
    # E.g., [0.5, 0.5] for 2D, or [0.5, 0.5, 0.5, 0.5] for 4D
    initial_steps = [0.5] * dimensions
    
    # Run the algorithm
    best_x, best_f, _, steps = hooke_jeeves(
        func=func, 
        x0=x0, 
        initial_steps=initial_steps,
        step_reduce_factor=2.0, 
        epsilon=1e-5
    )
    
    # Format the coordinates output cleanly
    coords = ", ".join([f"{val:.10f}" for val in best_x])
    
    print(f"--- {name} ---")
    print(f"Iterations: {steps}")
    print(f"Optimal Point: [{coords}]")
    print(f"Minimum Value: {best_f:.20f}\n")