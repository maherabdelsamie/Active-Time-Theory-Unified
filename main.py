import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Constants
C = 299792458  # Speed of light in m/s
h = 6.62607015e-34  # Planck's constant in m^2 kg / s
delta_t = 5  # Time delay for the GlobalTime calculations

class GlobalTime:
    def __init__(self, use_ath=True, particles=[], cesium_atoms=[]):
        self.phi_history = [0] * delta_t
        self.current_time = 0
        self.time_flow_rate = 1
        self.dt = 0.01
        self.time_flow_rates = []
        self.intrinsic_times = []  # List to store intrinsic time at each update
        self.use_ath = use_ath
        self.particles = particles
        self.cesium_atoms = cesium_atoms 


    def calculate_phi_derivative(self):
        if not self.use_ath:
            return 0
        phi_delayed = self.phi_history[-delta_t]
        s_t = np.random.normal(0, 1)
        states = np.array([p.state for p in self.particles])
        mean_state = np.mean(states, axis=0)
        variance = np.var(states, axis=0)
        phi_prime = s_t - 0.1 * phi_delayed + np.linalg.norm(mean_state) + np.linalg.norm(variance)
        return phi_prime

    def update_phi(self, energy_density):
        # Calculate the phi derivative using RK4 method
        def phi_derivative(phi, energy_density):
            
            return np.tanh(energy_density) - 0.1 * phi

        k1 = self.dt * phi_derivative(self.phi_history[-1], energy_density)
        k2 = self.dt * phi_derivative(self.phi_history[-1] + 0.5 * k1, energy_density)
        k3 = self.dt * phi_derivative(self.phi_history[-1] + 0.5 * k2, energy_density)
        k4 = self.dt * phi_derivative(self.phi_history[-1] + k3, energy_density)
        phi_update = (k1 + 2*k2 + 2*k3 + k4) / 6

        # Update phi_history with the new value
        new_phi = self.phi_history[-1] + phi_update
        self.phi_history.append(new_phi)
        if len(self.phi_history) > delta_t:
            self.phi_history.pop(0)

    def update_time_flow(self):
        current_phi = self.phi_history[-1]
        self.time_flow_rate = 1 + 0.05 * np.tanh(current_phi)
        self.time_flow_rates.append(self.time_flow_rate)

    def update_current_time(self):
        if self.use_ath:
            self.current_time += self.time_flow_rate * self.dt
        else:
            self.current_time += self.dt
        self.intrinsic_times.append(self.current_time)  # Track current time

    def calculate_energy_density(self):
        # Calculate energy density based on kinetic and potential energies, and energy exchange with the medium
        total_energy = sum(0.5 * p.mass * np.linalg.norm(p.velocity)**2 + p.potential_energy for p in self.particles)
        energy_exchange = sum(p.energy_exchange for p in self.particles)
        total_energy += energy_exchange
        volume = 1  # Assuming a unit volume for simplicity
        energy_density = total_energy / volume

        # Simulate the temporal modulation on cesium atoms based on energy density
        for atom in self.cesium_atoms:
            atom.calculate_transitions(self, energy_density)

        return energy_density


    def update_all_phi_influences(self):
        # This function updates the phi value for each particle based on their local energy density.
        for particle in self.particles:
            particle.update_velocity_due_to_phi(self)


class QuantumParticle:
    def __init__(self, position, velocity, mass=1.0):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.mass = mass
        self.dilated_times = []  # Tracks dilated time for each particle
        self.potential_energy = 0  
        self.energy_exchange = 0  

    def update_state(self, dt, global_time):
        # Calculate temporal aperture for ATH modified Lorentz factor calculation
        temporal_aperture = self.calculate_temporal_aperture(global_time)
        # Calculate ATH modified Lorentz factor
        lorentz_factor_ath = 1 / np.sqrt(1 - np.linalg.norm(self.velocity)**2 / C**2) * (1 + global_time.phi_history[-1] * temporal_aperture)
        effective_dt = dt * lorentz_factor_ath
        self.position += self.velocity * effective_dt  # Update position based on velocity
        
        # Append the effective dilated time to the particle's dilated_times list
        if isinstance(effective_dt, (int, float)):
            self.dilated_times.append(effective_dt)
        else:
            print(f"Non-numerical value detected for Particle's effective_dt: {effective_dt}")

    def calculate_temporal_aperture(self, global_time):
        # Calculate the influence of phi based on local conditions
        local_phi_influence = self.calculate_local_phi_influence(global_time)
        temporal_aperture = np.exp(-local_phi_influence)
        
        # Validate the output is numerical
        if not isinstance(temporal_aperture, (int, float)):
            print(f"Non-numerical temporal_aperture detected: {temporal_aperture}")
            return 1  # Return a default numerical value to avoid breaking the simulation
        return temporal_aperture

    def calculate_local_phi_influence(self, global_time):
        # Initialize an array to store influence values
        influence = np.zeros_like(self.velocity)
        for other_particle in global_time.particles:
            if other_particle is not self:
                direction = other_particle.position - self.position
                distance = np.linalg.norm(direction)
                if distance > 0:  # Avoid division by zero
                    influence += (direction / distance) * (1 / distance**2)
        # Normalize by the number of particles to get the average influence
        average_influence = influence / len(global_time.particles) if global_time.particles else 0
        
        if not isinstance(average_influence, (int, float, np.ndarray)):
            print(f"Non-numerical average_influence detected: {average_influence}")
            return 0  # Return a default numerical value to ensure numerical output
        return np.linalg.norm(average_influence)  # Ensure the output is a numerical value


    def interact(self, particles, global_time):
        
        for other_particle in particles:
            if other_particle is not self:
                direction = other_particle.position - self.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction /= distance  # Normalize direction vector
                    # Modulate the velocity update based on Phi
                    stochastic_influence = np.random.normal(loc=1, scale=global_time.phi_history[-1])
                    self.velocity += direction * stochastic_influence



    def calculate_dilated_time(self, dt, global_time):
        temporal_aperture = self.calculate_temporal_aperture(global_time)
        lorentz_factor = 1 / np.sqrt(1 - np.linalg.norm(self.velocity)**2 / C**2)
        lorentz_factor_ath = lorentz_factor * (1 + global_time.phi_history[-1] * temporal_aperture) if global_time.use_ath else lorentz_factor
        dilated_time = dt * lorentz_factor_ath
        self.dilated_times.append(dilated_time)

    def calculate_temporal_aperture(self, global_time):
        # Calculate the temporal aperture based on the local phi influence
        local_phi_influence = self.calculate_local_phi_influence(global_time)
        temporal_aperture = np.exp(-local_phi_influence)
        return temporal_aperture

    def update_velocity_due_to_phi(self, global_time):
        
        local_phi_effect = global_time.phi_history[-1] * self.calculate_local_phi_influence(global_time)
        self.velocity += local_phi_effect



class CesiumAtom:
    def __init__(self, initial_state):
        self.state = initial_state
        self.base_energy_levels = [
            0.0, 1.3859, 1.4546, 2.2981, 2.6986, 2.7210, 1.7977, 1.8098, 2.8007, 2.8060
        ]
        self.transition_frequencies = []

    def calculate_transitions(self, global_time, energy_density):
        # Accessing the phi value
        phi_current = global_time.phi_history[-1]

        # Assuming the initial state is 3 for this example
        initial_state_energy = self.base_energy_levels[self.state]

        # Implement a time-varying influence on the transition frequencies based on energy density and phi
        transition_energy = initial_state_energy * (1 + phi_current * energy_density)
        transition_frequency = transition_energy / h

        # Update the transition frequencies list with the new value
        self.transition_frequencies.append(transition_frequency)


# Initialize the cesium atoms with a range of different states
cesium_atoms = [CesiumAtom(initial_state=i % 10) for i in range(10)]

def simulate_temporal_network(num_nodes, num_steps, edge_prob=0.5):
    G = nx.DiGraph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                G.add_edge(i, j, weight=np.random.rand())

    particles = [QuantumParticle([1.0, 0.0, 0.0], np.random.uniform(-0.5, 0.5, 3) * C, mass=1.0) for _ in range(num_nodes)]
    cesium_atoms = [CesiumAtom(initial_state=np.random.randint(0, 4)) for _ in range(10)]
    global_time = GlobalTime(use_ath=True, particles=particles, cesium_atoms=cesium_atoms)

    for step in range(num_steps):
        energy_density = global_time.calculate_energy_density()
        global_time.update_phi(energy_density)  # Update global phi based on energy density

        global_time.update_all_phi_influences()  # Update particle velocities based on phi

        for particle in particles:
            particle.update_state(global_time.dt, global_time)
            particle.calculate_dilated_time(global_time.dt, global_time)
            particle.energy_exchange = np.random.uniform(-0.1, 0.1)  # Simulate energy exchange with the medium

        
        for atom in cesium_atoms:
            atom.calculate_transitions(global_time, energy_density)

        global_time.update_time_flow()
        global_time.update_current_time()

    return G, global_time, particles, cesium_atoms




# Plotting functions for the simulation results
def plot_time_flow_rates(global_time):
    plt.figure(figsize=(10, 6))
    plt.plot(global_time.time_flow_rates)
    plt.title('Time Flow Rates Over Simulation')
    plt.xlabel('Simulation Step')
    plt.ylabel('Time Flow Rate')
    plt.show()

def plot_particle_dilated_times(particles):
    plt.figure(figsize=(10, 6))
    for i, particle in enumerate(particles):
        plt.plot(particle.dilated_times, label=f'Particle {i}')
    plt.title('Particle Dilated Times Over Simulation')
    plt.xlabel('Simulation Step')
    plt.ylabel('Dilated Time')
    plt.legend()
    plt.show()

def plot_cesium_atom_transition_frequencies(cesium_atoms):
    plt.figure(figsize=(10, 6))
    for i, atom in enumerate(cesium_atoms):
        plt.plot(atom.transition_frequencies, label=f'Atom {i}')
    plt.title('Cesium Atom Transition Frequencies')
    plt.xlabel('Simulation Step')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_intrinsic_time_over_iterations(global_time):
    plt.figure(figsize=(10, 6))
    plt.plot(global_time.intrinsic_times)
    plt.title('Intrinsic Time Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Intrinsic Time')
    plt.show()

def plot_energy_exchange_over_simulation(particles):
    plt.figure(figsize=(10, 6))
    energy_exchange_data = [p.energy_exchange for p in particles]
    plt.plot(energy_exchange_data)
    plt.title('Energy Exchange Over Simulation')
    plt.xlabel('Simulation Step')
    plt.ylabel('Energy Exchange')
    plt.show()


# Normalize data for PCA
def normalize_data(particles):
    state_matrix = np.array([np.concatenate([p.position, p.velocity]) for p in particles])
    scaler = StandardScaler()
    normalized_matrix = scaler.fit_transform(state_matrix)
    return normalized_matrix

# Updated PCA Scree Plot function
def plot_pca_scree_plot(particles):
    normalized_matrix = normalize_data(particles)
    pca = PCA()
    pca.fit(normalized_matrix)
    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(var_exp) + 1), var_exp, alpha=0.6, align='center', label='Individual explained variance')
    plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('PCA Scree Plot')
    plt.legend(loc='best')
    plt.show()

def plot_node_degree_distribution(G):
    degrees = [degree for _, degree in G.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=20, density=True)
    plt.title('Node Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Density')
    plt.show()

def plot_spatial_coordinates_from_network(G):
    # Convert the graph into an adjacency matrix for PCA
    adjacency_matrix = nx.to_numpy_array(G)
    pca = PCA(n_components=2)  # Using 2 components for 2D visualization
    pca_result = pca.fit_transform(adjacency_matrix)

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    for i, (x, y) in enumerate(pca_result, start=1):
        plt.text(x, y, str(i), color='red')  # Optionally, label nodes
    plt.title('Spatial Coordinates from Temporal Network')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()    



# Execute the simulation with the updated QuantumParticle class
num_nodes = 200
num_steps = 500

# Execute the simulation
G, global_time, particles, cesium_atoms = simulate_temporal_network(num_nodes, num_steps)

# Apply interactions after the simulation has been executed
for particle in particles:
    particle.interact(particles, global_time)

# Run the plotting functions
plot_time_flow_rates(global_time)
plot_cesium_atom_transition_frequencies(cesium_atoms)
plot_intrinsic_time_over_iterations(global_time)
plot_energy_exchange_over_simulation(particles)

plot_pca_scree_plot(particles)
plot_node_degree_distribution(G)
plot_spatial_coordinates_from_network(G)

plot_particle_dilated_times(particles)
