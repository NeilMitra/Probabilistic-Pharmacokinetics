import numpy as np
import matplotlib.pyplot as plt

def simulate_stochastic_pk_model(D, rate_constants, t_max):
    # Initialize molecule counts with full compartment names
    n = {
        'Heart': D,     # Heart
        'Brain': 0,     # Brain
        'Kidneys': 0,   # Kidneys
        'Liver': 0,     # Liver
        'Lungs': 0      # Lungs
    }

    t = 0.0
    times = [t]
    n_history = {compartment: [count] for compartment, count in n.items()}

    # Reactions with full compartment names
    reactions = [
        {'from': 'Heart', 'to': 'Brain', 'rate': rate_constants['k_Heart_Brain']},
        {'from': 'Brain', 'to': 'Heart', 'rate': rate_constants['k_Brain_Heart']},
        {'from': 'Heart', 'to': 'Kidneys', 'rate': rate_constants['k_Heart_Kidneys']},
        {'from': 'Kidneys', 'to': 'Heart', 'rate': rate_constants['k_Kidneys_Heart']},
        {'from': 'Heart', 'to': 'Liver', 'rate': rate_constants['k_Heart_Liver']},
        {'from': 'Liver', 'to': 'Heart', 'rate': rate_constants['k_Liver_Heart']},
        {'from': 'Heart', 'to': 'Lungs', 'rate': rate_constants['k_Heart_Lungs']},
        {'from': 'Lungs', 'to': 'Heart', 'rate': rate_constants['k_Lungs_Heart']}
    ]

    # Main simulation loop
    while t < t_max:
        # Calculate propensities
        propensities = []
        for reaction in reactions:
            n_from = n[reaction['from']]
            rate = reaction['rate']
            propensity = rate * n_from
            propensities.append(propensity)

        a0 = sum(propensities)

        if a0 == 0:
            break  # No more reactions can occur

        # Generate random numbers
        r1 = np.random.random()
        r2 = np.random.random()

        # Time to next reaction
        dt = -np.log(r1) / a0
        t += dt

        # Determine which reaction occurs
        cumulative_propensity = 0.0
        threshold = r2 * a0
        for i, reaction in enumerate(reactions):
            cumulative_propensity += propensities[i]
            if cumulative_propensity > threshold:
                # Update molecule counts
                n[reaction['from']] -= 1
                n[reaction['to']] += 1
                break

        # Ensure molecule counts are non-negative
        for compartment in n:
            n[compartment] = max(n[compartment], 0)

        # Record time and molecule counts
        times.append(t)
        for compartment in n:
            n_history[compartment].append(n[compartment])

    return np.array(times), n_history

def plot_stochastic_results(times, n_history):
    plt.figure(figsize=(12, 8))
    for compartment, counts in n_history.items():
        plt.step(times, counts, where='post', label=compartment, linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Number of Molecules')
    plt.title('Probabilistic Pharmacokinetic Vital Organ Five-Compartment Model')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Initial drug dose
    D = 1000  # Number of drug molecules initially in the heart

    # Rate constants with updated keys
    rate_constants = {
        'k_Heart_Brain': 0.1,    # Heart to Brain
        'k_Brain_Heart': 0.05,   # Brain to Heart
        'k_Heart_Kidneys': 0.2,  # Heart to Kidneys
        'k_Kidneys_Heart': 0.1,  # Kidneys to Heart
        'k_Heart_Liver': 0.15,   # Heart to Liver
        'k_Liver_Heart': 0.1,    # Liver to Heart
        'k_Heart_Lungs': 0.25,   # Heart to Lungs
        'k_Lungs_Heart': 0.25    # Lungs to Heart
    }

    t_max = 50  # Maximum simulation time

    # Run stochastic simulation
    times, n_history = simulate_stochastic_pk_model(D, rate_constants, t_max)

    # Plot results
    plot_stochastic_results(times, n_history)

    # Print final molecule counts
    print("Final molecule counts:")
    for compartment, counts in n_history.items():
        print(f"{compartment}: {counts[-1]}")
