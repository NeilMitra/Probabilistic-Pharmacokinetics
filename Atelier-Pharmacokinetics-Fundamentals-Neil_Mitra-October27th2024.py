import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pk_model(state, t, k1, k2, V1, V2):
    """
    Two-compartment pharmacokinetic model differential equations
    
    Args:
        state: List containing [C1, C2]
        t: Time point
        k1, k2: Transfer rate constants
        V1, V2: Volumes of compartments
    Returns:
        List of derivatives [dC1/dt, dC2/dt]
    """
    C1, C2 = state
    
    # Differential equations from the model
    dC1_dt = (k1 * C2 * V2 - k2 * C1 * V1) / V1
    dC2_dt = (k2 * C1 * V1 - k1 * C2 * V2) / V2
    
    return [dC1_dt, dC2_dt]

def simulate_pk_model(D, V1, V2, k1, k2, t_max, num_points=1000):
    """
    Simulate the two-compartment model
    
    Args:
        D: Initial drug dose
        V1, V2: Volumes of compartments
        k1, k2: Transfer rate constants
        t_max: Maximum simulation time
        num_points: Number of time points to simulate
    """
    # Initial conditions
    C0 = D / V1  # Initial concentration in compartment 1
    initial_state = [C0, 0]  # [C1(0), C2(0)]
    
    # Time points
    t = np.linspace(0, t_max, num_points)
    
    # Solve ODE system
    solution = odeint(pk_model, initial_state, t, args=(k1, k2, V1, V2))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, solution[:, 0], 'r-', label='Heart (C1)', linewidth=2)
    plt.plot(t, solution[:, 1], 'b-', label='Lung (C2)', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (M)')
    plt.title('Two-Compartment Pharmacokinetic Model')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return t, solution

# Example usage
if __name__ == "__main__":
    # Model parameters
    D = 100  # Initial drug dose
    V1 = 1.0  # Volume of compartment 1 (Heart)
    V2 = 1.0  # Volume of compartment 2 (Lung)
    k1 = 0.5  # Transfer rate constant from lung to heart
    k2 = 0.5  # Transfer rate constant from heart to lung
    t_max = 20  # Maximum simulation time
    
    # Run simulation
    t, solution = simulate_pk_model(D, V1, V2, k1, k2, t_max)
    
    # Print some key results
    print(f"Maximum concentration in Heart: {max(solution[:, 0]):.2f}")
    print(f"Maximum concentration in Lung: {max(solution[:, 1]):.2f}")
