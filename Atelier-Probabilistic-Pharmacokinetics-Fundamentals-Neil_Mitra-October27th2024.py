import numpy as np
import matplotlib.pyplot as plt

def simulate_stochastic_pk_model(D, k1, k2, t_max):
    """
    Simulate the two-compartment pharmacokinetic model using the Gillespie SSA.

    Args:
        D: Initial drug dose (number of molecules)
        k1, k2: Transfer rate constants
        t_max: Maximum simulation time
    Returns:
        times: Array of time points
        n1_values: Array of molecule counts in compartment 1 (Heart)
        n2_values: Array of molecule counts in compartment 2 (Lung)
    """
    # Initialize variables
    n1 = D  # Initial number of molecules in compartment 1 (Heart)
    n2 = 0  # Initial number of molecules in compartment 2 (Lung)
    t = 0.0

    times = [t]
    n1_values = [n1]
    n2_values = [n2]

    # Main simulation loop
    while t < t_max:
        # Calculate propensities
        a1 = k2 * n1  # Propensity of Heart to Lung transfer
        a2 = k1 * n2  # Propensity of Lung to Heart transfer
        a0 = a1 + a2  # Total propensity

        if a0 == 0:
            break  # No more reactions can occur

        # Generate random numbers
        r1 = np.random.random()
        r2 = np.random.random()

        # Time to next reaction
        dt = -np.log(r1) / a0
        t += dt

        # Determine which reaction occurs
        if r2 * a0 < a1:
            # Reaction 1 occurs: Heart to Lung
            n1 -= 1
            n2 += 1
        else:
            # Reaction 2 occurs: Lung to Heart
            n1 += 1
            n2 -= 1

        # Ensure molecule counts are non-negative
        n1 = max(n1, 0)
        n2 = max(n2, 0)

        # Record time and molecule counts
        times.append(t)
        n1_values.append(n1)
        n2_values.append(n2)

    return np.array(times), np.array(n1_values), np.array(n2_values)

def plot_stochastic_results(times, n1_values, n2_values):
    """
    Plot the results of the stochastic simulation.

    Args:
        times: Array of time points
        n1_values: Array of molecule counts in compartment 1 (Heart)
        n2_values: Array of molecule counts in compartment 2 (Lung)
    """
    plt.figure(figsize=(10, 6))
    plt.step(times, n1_values, where='post', label='Heart (n1)', linewidth=2)
    plt.step(times, n2_values, where='post', label='Lung (n2)', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Molecules')
    plt.title('Probabilistic Two-Compartment Pharmacokinetic Model')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Model parameters
    D = 100  # Initial drug dose (number of molecules)
    k1 = 0.5  # Transfer rate constant from lung to heart
    k2 = 0.5  # Transfer rate constant from heart to lung
    t_max = 20  # Maximum simulation time

    # Run stochastic simulation
    times, n1_values, n2_values = simulate_stochastic_pk_model(D, k1, k2, t_max)

    # Plot results
    plot_stochastic_results(times, n1_values, n2_values)

    # Print some key results
    print(f"Final molecule count in Heart: {n1_values[-1]}")
    print(f"Final molecule count in Lung: {n2_values[-1]}")
