import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import torch

from main import bicycle_dynamics_perfect_disturbed

# -------------------------------
# Parameters and Dynamics Functions
# -------------------------------
N = 100           # prediction horizon (number of time steps)
dt = 0.1         # time step size
L = 2.5          # wheelbase of the bicycle model

# Nominal dynamics function in CasADi syntax
def nominal_dynamics(xk, uk, dt):
    x_val, y_val, theta_val, v_val = xk[0], xk[1], xk[2], xk[3]
    delta_val, a_val = uk[0], uk[1]
    x_next = x_val + dt * v_val * ca.cos(theta_val)
    y_next = y_val + dt * v_val * ca.sin(theta_val)
    theta_next = theta_val + dt * (v_val / L) * ca.tan(delta_val)
    v_next = v_val + dt * a_val
    return ca.vertcat(x_next, y_next, theta_next, v_next)

# -------------------------------
# Residual Correction Functions
# -------------------------------
# Placeholder: returns zero correction.
def residual_correction(xk, uk):
    return ca.DM.zeros(4)

# --- Integration of a Neural Network Residual Correction ---
try:
    nn_model = torch.load(".\\residual_model.pth", map_location=torch.device("cpu"))
    nn_model.eval()

    def nn_residual_correction(xk, uk):
        xk_np = np.array(xk.full()).flatten()  # convert to numpy
        uk_np = np.array(uk.full()).flatten()
        inp = torch.tensor(np.concatenate([xk_np, uk_np]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = nn_model(inp).numpy()  # expected shape (1,4)
        return ca.DM(pred.T)
except Exception as e:
    print("Warning: Failed to load NN model; using zero residual correction.")
    def nn_residual_correction(xk, uk):
        return ca.DM.zeros(4)

USE_NN_CORRECTION = True  # Set to True if you want to use the NN correction
def used_residual_correction(xk, uk):
    if USE_NN_CORRECTION:
        return nn_residual_correction(xk, uk)
    else:
        return residual_correction(xk, uk)

# -------------------------------
# MPC Problem Setup Function (Open-Loop Solver)
# -------------------------------
def solve_mpc(x0_val, X_ref_val):
    opti = ca.Opti()
    X = opti.variable(4, N+1)   # states: 4 x (N+1)
    U = opti.variable(2, N)     # controls: 2 x N
    
    x0 = opti.parameter(4, 1)
    X_ref = opti.parameter(4, N+1)
    
    for k in range(N):
        x_next = nominal_dynamics(X[:, k], U[:, k], dt) + used_residual_correction(X[:, k], U[:, k])
        opti.subject_to(X[:, k+1] == x_next)
    
    opti.subject_to(X[:, 0] == x0)
    
    # old cost function (commented out)
    # cost = 0
    # for k in range(N):
    #     cost += ca.sumsqr(X[:, k] - X_ref[:, k])
    #     cost += 0.01 * ca.sumsqr(U[:, k])
    # cost += ca.sumsqr(X[:, N] - X_ref[:, N])
    # opti.minimize(cost)

    # new cost function with weights
    # Define weight matrices as NumPy arrays
    Q = np.diag([1.0, 1.0, 0.1, 0.1])    # State tracking weight
    # Q = np.diag([2.0, 2.0, 0.5, 0.5])
    # Q = np.diag([4.0, 4.0, 0.5, 0.5]) # best
    # Q = np.diag([4.0, 4.0, 0.5, 2.0])
    # Q = np.diag([3.0, 3.0, 1.0, 1.0])
    R = np.diag([0.01, 0.01])             # Control effort weight
    # R = np.diag([0.02, 0.02])
    # R = np.diag([0.03, 0.03]) # best
    # R = np.diag([0.05, 0.05])
    # R = np.diag([0.03, 0.03])
    P = np.diag([10.0, 10.0, 1.0, 1.0])   # Terminal cost weight
    # P = np.diag([20.0, 20.0, 5.0,15.0]) # example heavier on v if final speed must be 0
    # P = np.diag([30.0, 30.0, 10.0, 20.0]) 
    # P = np.diag([50.0, 50.0, 5.0, 50.0])
    # P = np.diag([50.0, 50.0, 5.0, 80.0])
    # P = np.diag([30.0, 30.0, 5.0, 30.0])

    # Q = np.diag([1.0, 1.0, 0.1, 0.1])  
    # R = np.diag([0.01, 0.01]) 
    # P = np.diag([100.0, 100.0, 10.0, 100.0])

    # Build the cost function
    cost = 0
    for k in range(N):
        # Compute the state error and add the quadratic cost
        x_diff = X[:, k] - X_ref[:, k]
        u_k = U[:, k]
        cost += ca.mtimes([x_diff.T, Q, x_diff]) + ca.mtimes([u_k.T, R, u_k])
        
    # Terminal cost for the final state at k = N
    xN_diff = X[:, N] - X_ref[:, N]
    cost += ca.mtimes([xN_diff.T, P, xN_diff])

    opti.minimize(cost)
    
    opti.subject_to(opti.bounded(-0.2, U[0, :], 0.2))
    opti.subject_to(opti.bounded(-1, U[1, :], 1))
    
    opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt", opts)
    
    opti.set_value(x0, x0_val)
    opti.set_value(X_ref, X_ref_val)
    
    sol = opti.solve()
    X_opt = sol.value(X)
    U_opt = sol.value(U)
    return X_opt, U_opt

# -------------------------------
# Open-Loop Demonstration Function
# -------------------------------
def open_loop_demo():
    x0_val = np.array([0.0, 0.0, 0.0, 0.0]).reshape(4, 1)
    target_state = np.array([7.0, 7.0, 0.0, 0.0]).reshape(4, 1)
    X_ref_val = np.tile(target_state, (1, N+1))
    
    X_opt, U_opt = solve_mpc(x0_val, X_ref_val)
    
    print("Open-loop optimal state trajectory:\n", X_opt)
    print("Open-loop optimal control sequence:\n", U_opt)
    
    plt.figure(figsize=(8,6))
    plt.plot(X_opt[0, :], X_opt[1, :], 'b-o', label='Planned Trajectory')
    plt.scatter(x0_val[0], x0_val[1], c='green', marker='o', s=100, label='Start')
    plt.scatter(target_state[0], target_state[1], c='red', marker='*', s=150, label='Target')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Open-loop MPC Trajectory')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# Closed-Loop Simulation Function
# -------------------------------
def closed_loop_simulation(sim_time=10):
    num_steps = int(sim_time / dt)
    # Initialize state as a (4,1) vector.
    state = np.array([0.0, 0.0, 0.0, 0.0]).reshape(4, 1)
    
    target_state = np.array([7.0, 7.0, 0.0, 0.0]).reshape(4, 1)
    X_ref_val = np.tile(target_state, (1, N+1))
    
    state_history = [state]
    control_history = []
    
    for t in range(num_steps):
        # Ensure state is a (4,1) column vector.
        if state.shape != (4, 1):
            if state.size == 4:
                state = state.reshape(4, 1)
            elif state.shape[0] == 4 and state.shape[1] > 1:
                # print(f"Warning: state has shape {state.shape}. Taking last column as the current state.")
                state = state[:, -1].reshape(4, 1)
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
        
        # Solve MPC using the current state as the initial condition.
        X_opt, U_opt = solve_mpc(state, X_ref_val)
        u_apply = U_opt[:, 0].reshape(2, 1)
        control_history.append(u_apply)
        
        # Update state using disturbed dynamics.
        _, state = bicycle_dynamics_perfect_disturbed(state, u_apply, dt)
        state_history.append(state)
        
    # Concatenate the state history (each state should be 4x1; hstack yields a 4 x num_steps+1 matrix)
    state_history = np.hstack(state_history)
    control_history = np.hstack(control_history)

    # position RMSE
    x = state_history[0,:]
    y = state_history[1,:]
    xref = 7.0; yref = 7.0
    e = np.sqrt((x - xref)**2 + (y - yref)**2)
    rmse = np.sqrt(np.mean(e**2))
    print(f"Position RMSE = {rmse:.3f} m")
    
    plt.figure(figsize=(8,6))
    plt.plot(state_history[0, :], state_history[1, :], '-o', label='Closed-loop Trajectory')
    # Highlight the start and target points.
    plt.scatter(state_history[0, 0], state_history[1, 0], c='green', marker='o', s=100, label='Start')
    plt.scatter(target_state[0], target_state[1], c='red', marker='*', s=150, label='Target')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Closed-loop MPC Trajectory')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return state_history, control_history

# -------------------------------
# Main Code: Execute Demonstrations
# -------------------------------
if __name__ == "__main__":
    print("Running open-loop MPC demo:")
    open_loop_demo()
    
    print("\nRunning closed-loop MPC simulation:")
    closed_loop_simulation(sim_time=10)


