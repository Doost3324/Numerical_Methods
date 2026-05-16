from main import *
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Plotting the System to find an initial guess ---

def plot_system():
    # Create a grid of points
    x1_vals = np.linspace(-3, 3, 400)
    x2_vals = np.linspace(-3, 4, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    # Define our two equations (set to equal 0)
    Eq1 = X1**2 + X2**2 - 4
    Eq2 = X2 - X1**2
    
    plt.figure(figsize=(8, 6))
    
    # We use contour plots at level [0] to draw the lines where the equations are true
    cs1 = plt.contour(X1, X2, Eq1, levels=[0], colors='blue')
    cs2 = plt.contour(X1, X2, Eq2, levels=[0], colors='red')
    
    # Adding a legend (contour plots require a slight workaround for legends)
    lines1, _ = cs1.legend_elements()
    lines2, _ = cs2.legend_elements()
    plt.legend([lines1[0], lines2[0]], ['$x_1^2 + x_2^2 - 4 = 0$', '$x_2 - x_1^2 = 0$'])
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Visualizing Intersections for Initial Approximation')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()

# Run the plot
plot_system()


# --- 2. Solving the System using Hooke-Jeeves ---

# Define the Objective Function Phi(X)
def system_objective_function(x):
    eq1_val = x[0]**2 + x[1]**2 - 4.0
    eq2_val = x[1] - x[0]**2
    
    # Sum of squares
    return (eq1_val)**2 + (eq2_val)**2

# From looking at the plot generated above, we can see an intersection 
# happening in the top right quadrant around x1=1.2, x2=1.5. 
# Let's use that as our initial guess X0.
x0_system = [1.2, 1.5]

# Using the algorithm from Step 1
best_x_sys, best_f_sys, trajectory_sys, steps_sys = hooke_jeeves(
    func=system_objective_function, 
    x0=x0_system, 
    initial_steps=[0.5, 0.5],
    step_reduce_factor=2.0, 
    epsilon=1e-5
)

print("\n--- Solving System of Non-Linear Equations ---")
print(f"Optimization finished in {steps_sys} iterations.")
print(f"Intersection found at (x1, x2): [{best_x_sys[0]:.5f}, {best_x_sys[1]:.5f}]")
print(f"Objective Function Value Phi(X): {best_f_sys:.15f}")

def save_trajectory_to_file(filename, trajectory, total_steps):
    """Saves the optimization trajectory coordinates to a text file."""
    try:
        # Using 'w' mode to write (will overwrite the file if it already exists)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("Hooke-Jeeves Descent Trajectory\n")
            file.write("===============================\n")
            file.write(f"Total iterations required: {total_steps}\n\n")
            
            file.write("Coordinates at each base point update:\n")
            file.write("-" * 40 + "\n")
            
            # Loop through the recorded trajectory and write each point
            for i, point in enumerate(trajectory):
                # Format to 5 decimal places for clean reading
                file.write(f"Base Point {i:2d}: x1 = {point[0]:.5f}, x2 = {point[1]:.5f}\n")
                
            file.write("-" * 40 + "\n")
            file.write("End of trajectory.\n")
                
    except Exception as e:
        print(f"\nError writing to file: {e}")

# Call the function using the results from our system equation solve
save_trajectory_to_file("trajectory_output.txt", trajectory_sys, steps_sys)