import numpy as np

def exploratory_search(func, base_point, step_sizes):
    n = len(base_point)
    current_point = np.copy(base_point)
    best_val = func(current_point)
    
    for i in range(n):
        # Try stepping forward
        current_point[i] += step_sizes[i]
        if func(current_point) < best_val:
            best_val = func(current_point)
        else:
            # Try stepping backward
            current_point[i] -= 2 * step_sizes[i]
            if func(current_point) < best_val:
                best_val = func(current_point)
            else:
                # Revert to original coordinate if neither improved
                current_point[i] += step_sizes[i]
                
    return current_point, best_val

def hooke_jeeves(func, x0, initial_steps, step_reduce_factor=2.0, epsilon=1e-5, max_iter=2000):
    x_base = np.array(x0, dtype=float)
    step_sizes = np.array(initial_steps, dtype=float)
    
    trajectory = [np.copy(x_base)] # To satisfy Requirement 5 later
    steps_taken = 0
    
    while np.max(step_sizes) > epsilon and steps_taken < max_iter:
        steps_taken += 1
        
        # 1. Exploratory search around the base point
        x_new, f_new = exploratory_search(func, x_base, step_sizes)
        
        # If exploratory search is successful
        if f_new < func(x_base):
            while True:
                steps_taken += 1
                # 2. Pattern move
                x_pattern = x_base + 2.0 * (x_new - x_base)
                
                # Update base point
                x_base = np.copy(x_new)
                trajectory.append(np.copy(x_base))
                
                # Exploratory search around the pattern point
                x_new_pattern, f_new_pattern = exploratory_search(func, x_pattern, step_sizes)
                
                # If pattern search fails to improve, break out and reduce step size
                if f_new_pattern >= func(x_base):
                    break
                
                x_new = np.copy(x_new_pattern)
        else:
            # If exploratory search fails, reduce the step size
            step_sizes /= step_reduce_factor
            
    return x_base, func(x_base), trajectory, steps_taken

