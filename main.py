import numpy as np
import matplotlib.pyplot as plt

'''
Author: Reem Al Mazroa
'''
# Parameters for the bicycle model
L = 2.5  # wheelbase, in meters

# Bicycle model definition
def bicycle_dynamics_perfect(state, control, dt):
    """
    state: [x, y, theta, v]
    control: [delta, a]
    dt: time step
    """
    x, y, theta, v = state
    delta, a = control
    
    # Update equations (using Euler integration for simplicity)
    x_next = x + dt * v * np.cos(theta)
    y_next = y + dt * v * np.sin(theta)
    theta_next = theta + dt * (v / L) * np.tan(delta)
    v_next = v + dt * a
    
    return np.array([x_next, y_next, theta_next, v_next])

# bicycle dynamics with disturbances
def bicycle_dynamics_perfect_disturbed(state, control, dt):
    """
    state: [x, y, theta, v]
    control: [delta, a]
    dt: time step
    Returns the next state using "true" dynamics with disturbance.
    """
    # Nominal dynamics as before
    nominal_state = bicycle_dynamics_perfect(state, control, dt)
    
    # Define disturbances
    disturbance = np.array([0.05 * np.random.randn(), # x
                            0.05 * np.random.randn(), # y
                            0.02 * np.random.randn(), # theta
                            0.1 * np.random.randn()]) # v
    # this^ is actually exactly equal to the residuals we are trying to predict, but we do it this way for clarity and modular design
    
    return nominal_state, nominal_state + disturbance

def example_simulation():
    # Simulation parameters
    dt = 0.1          # time step (seconds)
    t_end = 10        # total simulation time (seconds)
    N = int(t_end / dt)

    # Initial state: x, y, theta, v
    state = np.array([0.0, 0.0, np.pi/4, 5.0])  # Starting at (0,0), 45° angle, 5 m/s

    # Choose a constant control input for testing:
    # For example, a small constant steering angle and zero acceleration.
    control = np.array([0.1, 0.0])  # steering angle = 0.1 rad, acceleration = 0 m/s^2

    # Simulate the system
    trajectory = [state.copy()]
    for _ in range(N):
        state = bicycle_dynamics_perfect(state, control, dt)
        trajectory.append(state.copy())

    trajectory = np.array(trajectory)

    # Plot the trajectory (x vs. y)
    plt.figure(figsize=(8,6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Bicycle Model Trajectory')
    plt.legend()
    plt.grid(True)
    plt.show()

def check_data_generation():
    # Load the dataset files
    inputs = np.load("mpc_inputs.npy")
    targets = np.load("mpc_targets.npy")

    # Print the shapes
    print("Inputs shape:", inputs.shape)
    print("Targets shape:", targets.shape)

    # Inspect the first few samples
    print("First 5 input samples:\n", inputs[:5])
    print("First 5 target samples:\n", targets[:5])

#example_simulation() # -> looks good
check_data_generation() # -> looks good
