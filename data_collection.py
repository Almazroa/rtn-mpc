from main import bicycle_dynamics_perfect_disturbed
import numpy as np

'''
Author: Reem Al Mazroa
'''

'''
IDEA: Data collection
- run the dynamics model then add noise to the output (next state); represents "true" dynamics
- take difference between the "true" dynamics and the nominal dynamics (disturbed vs perfect); represents the residual
- pair residual with original (current) state and control input to create a dataset for training a neural network
- In training, the neural network learns to predict this residual error from the nominal prediction, based solely on the state and control input without the added disturbance

"In real systems, the control inputs can lead to non-ideal outcomes because of external factors (e.g., wind, uneven terrain) and internal factors (e.g., sensor noise, friction). 
By training the network on these residuals, you essentially teach it to correct the nominal model's prediction to account for those imperfections."
'''

# Run simulation for a given control sequence and initial state
def run_simulation(initial_state, control_sequence, dt):
    """
    Simulate both nominal and true dynamics over a sequence of control inputs.
    Returns the nominal trajectory, true trajectory, and residuals.
    """
    nominal_traj = [initial_state.copy()]
    true_traj = [initial_state.copy()]
    residuals = []  # residual for each time step (true - nominal)

    state_nom = initial_state.copy()
    state_true = initial_state.copy()

    for control in control_sequence:
        state_nom_next, state_true_next = bicycle_dynamics_perfect_disturbed(state_nom, control, dt)
        # Calculate residual (difference between true and nominal)
        residual = state_true_next - state_nom_next

        nominal_traj.append(state_nom_next.copy())
        true_traj.append(state_true_next.copy())
        residuals.append(residual.copy())

        state_nom = state_nom_next
        state_true = state_true_next

    return np.array(nominal_traj), np.array(true_traj), np.array(residuals)

'''
this sim runs 100 trials. Each with a random initial state and a random control sequence for 100 time steps
at each time step, the simulation computes the nominal next state, the disturbed (true) state, and then calculates the residual.
all of this is stored in a dataset for training a neural network
100 trials * 100 time steps = 10,000 samples

'''

# Parameters for the simulation
dt = 0.1         # time step in seconds
T = 10           # total simulation time in seconds
N = int(T/dt)    # number of simulation steps

# Generate a dataset over multiple simulation runs (trials)
num_trials = 100
data_list = []

for trial in range(num_trials):
    # Randomly choose an initial state: [x, y, theta, v]
    initial_state = np.array([
        np.random.uniform(-1, 1),       # x position
        np.random.uniform(-1, 1),       # y position
        np.random.uniform(-np.pi, np.pi),# orientation (theta)
        np.random.uniform(0, 10)         # speed (v)
    ])

    # Generate a random control sequence for the trial
    control_sequence = []
    for _ in range(N):
        delta = np.random.uniform(-0.2, 0.2)  # steering angle (radians)
        a = np.random.uniform(-1, 1)          # acceleration (m/s^2)
        control_sequence.append(np.array([delta, a]))
    control_sequence = np.array(control_sequence)

    # Run the simulation
    nominal_traj, true_traj, residuals = run_simulation(initial_state, control_sequence, dt)
    
    # For training, we pair the input (nominal state and control) with the residual.
    # Since residuals are computed at each step, we have N residuals corresponding to steps 1 to N.
    for t in range(N):
        # Input features: [state at time t, control at time t]
        input_features = np.concatenate((nominal_traj[t], control_sequence[t]))
        target_residual = residuals[t]  # residual corresponding to moving to t+1
        data_list.append((input_features, target_residual))

# Convert list to arrays for training
inputs = np.array([d[0] for d in data_list])   # Shape: (num_trials * N, 6)
targets = np.array([d[1] for d in data_list])   # Shape: (num_trials * N, 4)

print("Dataset shapes:", "Inputs:", inputs.shape, "Targets:", targets.shape)

# Optionally, save the dataset to disk
np.save("mpc_inputs.npy", inputs)
np.save("mpc_targets.npy", targets)