import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. Global Parameters and Constants ---
# Consumer action values (kWh) - Discretized for population game
# Represents selling energy, low consumption, medium consumption, high consumption
ACTION_VALUES = np.array([-5.0, 0.0, 5.0, 15.0]) 

# Price options for electricity (unit/kWh)
PRICE_OPTIONS = np.array([0.1, 0.5, 1.0])

# Simulation parameters
TIME_STEPS = 200  # Number of simulation steps
DT = 0.1          # Time step for replicator dynamics

# Aggregator parameters (from paper, simplified)
RHO_DA = 0.2      # Price of electricity paid by aggregator in day-ahead market
A_AGGREGATOR = 100 # Energy purchased by the aggregator in day-ahead market (kWh)
RHO_RT = 1.5      # Price of electricity paid by aggregator in real-time market (penalty for exceeding A_AGGREGATOR)
THETA_AGGREGATOR = 50 # Coefficient of grid awareness for the aggregator (penalty for voltage deviation)

# Simplified grid parameters
INITIAL_VOLTAGE_PU = 1.0 # Initial per-unit voltage
S_II_AVG = 0.005 # Average diagonal element of voltage sensitivity matrix (simplified)
# Note: For a detailed grid, S(i,i) would be specific to each node. Here, it's an average.

# Initial average previous action for voltage calculation (simplified)
PREV_TOTAL_CONSUMPTION_AVG = 50.0 # Placeholder for initial aggregate previous action

# Dynamic renewable generation parameters
# Amplitude controls the magnitude of seasonal variation while NOISE_STD adds
# random fluctuations to capture weather uncertainty.
R_DYNAMIC_AMPLITUDE = 0.3  # 30% seasonal variation
R_NOISE_STD = 0.1          # Standard deviation of multiplicative noise

# Aggregator risk aversion for price volatility
AGG_RISK_COEFF = 0.05


# --- 2. Consumer Model Functions (Prospect Theory) ---

def probability_weighting_function(p, alpha):
    """
    Implements Prelec's probability weighting function (Equation 2 from paper).
    Args:
        p (float): Objective probability.
        alpha (float): Coefficient of rationality (0 <= alpha <= 1).
    Returns:
        float: Subjective probability.
    """
    if p == 0: return 0.0
    if p == 1: return 1.0
    return np.exp(-(-np.log(p))**alpha)

def individual_payoff(a_i, rho, beta_i, R_i, gamma_i, v_i, S_ii, a_hat_i):
    """
    Calculates the payoff for an individual active consumer (Equation 3 from paper).
    Args:
        a_i (float): Action of the i-th active consumer (energy purchased/sold).
        rho (float): Price of electricity.
        beta_i (float): Comfort coefficient.
        R_i (float): Renewable generation available to the consumer.
        gamma_i (float): Coefficient of grid awareness.
        v_i (float): Distribution grid voltage at node i.
        S_ii (float): i-th diagonal element of the voltage sensitivity matrix.
        a_hat_i (float): Previous action of the i-th active consumer.
    Returns:
        float: Individual consumer's payoff.
    """
    # Ensure a_i + R_i + 1 > 0 for log
    comfort_term = beta_i * np.log(max(a_i + R_i + 1, 1e-9)) # Add small epsilon to avoid log(0)
    
    cost_term = rho * a_i
    
    # Voltage deviation term
    voltage_deviation = (v_i + S_ii * (a_hat_i - a_i) - 1)**2
    grid_awareness_term = gamma_i * voltage_deviation
    
    return comfort_term - cost_term - grid_awareness_term

def perceived_payoff(a_i, price_distribution, price_options, consumer_params):
    """
    Calculates the perceived payoff for an active consumer according to Prospect Theory (Equation 5).
    Args:
        a_i (float): Action of the consumer.
        price_distribution (np.array): Probability distribution over price_options.
        price_options (np.array): Possible electricity prices.
        consumer_params (dict): Dictionary with 'beta', 'gamma', 'alpha', 'R_i', 'v_i', 'S_ii', 'a_hat_i'.
    Returns:
        float: Perceived payoff.
    """
    total_perceived_payoff = 0.0
    for j, p_obj in enumerate(price_options):
        omega_p = probability_weighting_function(price_distribution[j], consumer_params['alpha'])
        
        # Calculate individual payoff for this price option
        u_i = individual_payoff(
            a_i, p_obj, consumer_params['beta'], consumer_params['R_i'],
            consumer_params['gamma'], consumer_params['v_i'], consumer_params['S_ii'],
            consumer_params['a_hat_i']
        )
        total_perceived_payoff += u_i * omega_p
    return total_perceived_payoff

# --- 3. Aggregator Model Functions ---

def calculate_voltage_deviation(total_current_consumption, S_ii_avg, initial_voltage_pu, prev_total_consumption):
    """
    Simplified model for calculating aggregate voltage deviation.
    This replaces the detailed grid simulation.
    Args:
        total_current_consumption (float): Sum of all consumer actions.
        S_ii_avg (float): Average voltage sensitivity.
        initial_voltage_pu (float): Reference voltage.
        prev_total_consumption (float): Total consumption in previous step.
    Returns:
        float: Absolute voltage deviation from 1 p.u.
    """
    # A simplified linear model: voltage changes based on deviation from previous total consumption
    # and an average sensitivity. The '1' in the paper's formula refers to 1 p.u.
    # We are calculating the deviation from 1 p.u.
    # Simplified: V_new = V_initial + S_avg * (delta_a)
    # Deviation = V_new - 1
    voltage_change = S_ii_avg * (total_current_consumption - prev_total_consumption)
    current_voltage = initial_voltage_pu + voltage_change
    return np.abs(current_voltage - 1.0)

def aggregator_profit_objective(price_probabilities, consumer_population_state, 
                                 aggregator_params, price_options, 
                                 S_ii_avg, initial_voltage_pu, prev_total_consumption):
    """
    Objective function for the aggregator's optimization.
    The aggregator wants to maximize its profit, so we return the negative profit for minimization.
    Args:
        price_probabilities (np.array): Current probability distribution over price_options.
        consumer_population_state (dict): Current state of the consumer population
                                         {'proportions': 2D array, 'consumer_types_params': list of dicts}.
        aggregator_params (dict): Aggregator's fixed parameters.
        price_options (np.array): Possible electricity prices.
        S_ii_avg (float): Average voltage sensitivity.
        initial_voltage_pu (float): Reference voltage.
        prev_total_consumption (float): Total consumption in previous step.
    Returns:
        float: Negative of the aggregator's profit.
    """
    current_proportions = consumer_population_state['proportions']
    consumer_types_params = consumer_population_state['consumer_types_params']
    
    # Calculate the average action of the entire population given the current price_probabilities
    # For each consumer type, we assume they pick the action that maximizes their perceived payoff
    # based on the aggregator's announced price_probabilities.
    # This is a simplification: in true replicator dynamics, the population *evolves* towards
    # better strategies, it doesn't instantly pick the best.
    # However, for the aggregator's optimization, it needs to predict consumer response.
    
    # To make this tractable, we'll assume consumers react to the 'perceived price of electricity' (P_hat)
    # as defined in the paper (Equation 14) when calculating their best response for the aggregator's objective.
    
    # Calculate perceived price P_hat for the current price_probabilities
    sum_rho_omega_p = 0.0
    sum_omega_p = 0.0
    for j, p_obj in enumerate(price_options):
        # Use an average alpha for P_hat calculation, or assume aggregator knows average rationality
        # For simplicity, let's use a fixed alpha (e.g., 0.7) for this P_hat calculation.
        # In a more complex model, aggregator might estimate average alpha.
        avg_alpha_for_phat = 0.7 
        omega_p = probability_weighting_function(price_probabilities[j], avg_alpha_for_phat)
        sum_rho_omega_p += p_obj * omega_p
        sum_omega_p += omega_p
    
    P_hat = sum_rho_omega_p / (sum_omega_p + 1e-9) if sum_omega_p > 1e-9 else 0.0

    # Calculate average consumer action based on P_hat for the aggregator's prediction
    # This is a simplification: the aggregator needs to predict the *aggregate* consumer action
    # that would result from its chosen price_probabilities.
    
    # A more robust way: for each consumer type and each action, calculate the payoff.
    # Then, the aggregator can assume consumers will shift towards actions with higher payoffs.
    # For this objective, let's assume the aggregator predicts the *average* action based on the current
    # population distribution and the perceived price.
    
    total_consumption_predicted = 0.0
    for type_idx, params in enumerate(consumer_types_params):
        # Calculate the best response action for this consumer type given the P_hat
        # This is a simplification for the aggregator's prediction.
        # The actual replicator dynamics will use the full perceived_payoff.
        
        # We need to re-derive the best response for the aggregator's prediction,
        # or use a simplified model for how total_consumption_predicted changes.
        
        # Let's simplify: the aggregator calculates the expected action based on the
        # current population distribution and the *expected value* of the price.
        expected_price = np.sum(price_probabilities * price_options)
        
        # This part is tricky. The paper's best response (Theorem 2) is complex.
        # For the aggregator's objective, we need a way to relate price_probabilities
        # to the resulting aggregate_action.
        
        # Let's use a simple linear relationship for aggregator's prediction:
        # Higher expected price -> lower consumption
        # This is a heuristic for the aggregator's internal model.
        # This is a major simplification from the paper's Theorem 2.
        
        # For the purpose of this population game, let's assume the aggregator's
        # prediction of total consumption is simply the sum of (proportion * action)
        # weighted by the current price probabilities.
        
        # A better way for the aggregator to predict:
        # It calculates the perceived payoff for each action for each consumer type.
        # Then, it assumes the population will shift towards actions with higher payoffs.
        # For the optimization, it needs a continuous function of price_probabilities.
        
        # Let's use a simpler approach for the aggregator's prediction of total_consumption:
        # It assumes that for a given price distribution, consumers will on average
        # choose actions that are 'optimal' given their type and the *expected* price.
        # This is still a simplification.
        
        predicted_actions_for_type = []
        for action_val in ACTION_VALUES:
            pp = perceived_payoff(action_val, price_probabilities, price_options, params)
            predicted_actions_for_type.append(pp)
        
        # Convert payoffs to probabilities (e.g., using softmax) to get a 'predicted distribution'
        # Then sum (predicted_prob * action_val)
        # FIX: Convert predicted_actions_for_type to a NumPy array
        predicted_actions_for_type_np = np.array(predicted_actions_for_type)
        predicted_probs_for_type = np.exp(predicted_actions_for_type_np / 10.0) # Scale for softmax
        predicted_probs_for_type /= np.sum(predicted_probs_for_type)
        
        total_consumption_predicted += np.sum(predicted_probs_for_type * ACTION_VALUES) * np.sum(current_proportions[type_idx])
        # The np.sum(current_proportions[type_idx]) is 1, so it's just the average action for the type.
        
    
    # Calculate revenue from consumers
    revenue = np.sum(price_probabilities * price_options) * total_consumption_predicted
    
    # Cost from day-ahead market
    cost_da = aggregator_params['RHO_DA'] * aggregator_params['A_AGGREGATOR']
    
    # Cost from real-time market (if total consumption exceeds day-ahead purchase)
    additional_energy_cost = max(0, total_consumption_predicted - aggregator_params['A_AGGREGATOR']) * aggregator_params['RHO_RT']
    
    # Voltage penalty
    voltage_deviation = calculate_voltage_deviation(
        total_consumption_predicted, S_II_AVG, initial_voltage_pu, prev_total_consumption
    )
    voltage_penalty = aggregator_params['THETA_AGGREGATOR'] * voltage_deviation**2

    # Risk aversion: penalize high variance in price distribution
    risk_penalty = aggregator_params.get('RISK_COEFF', 0.0) * np.var(price_probabilities)

    profit = revenue - cost_da - additional_energy_cost - voltage_penalty - risk_penalty
    
    return -profit # Minimize negative profit (maximize profit)


# --- 4. Simulation Class ---

class PopulationGameSimulation:
    def __init__(self, consumer_types_params, initial_proportions, action_values, price_options, aggregator_params):
        """
        Initializes the population game simulation.
        Args:
            consumer_types_params (list of dict): List of dictionaries, each defining a consumer type's
                                                 'beta', 'gamma', 'alpha', 'R_i', 'v_i', 'S_ii', 'a_hat_i'.
            initial_proportions (np.array): 2D array (num_consumer_types x num_actions)
                                            of initial proportions for each action within each type.
            action_values (np.array): Discrete action values.
            price_options (np.array): Possible electricity prices.
            aggregator_params (dict): Aggregator's fixed parameters.
        """
        self.consumer_types_params = consumer_types_params
        self.proportions = initial_proportions # Current proportions
        self.action_values = action_values
        self.price_options = price_options
        self.aggregator_params = aggregator_params
        
        self.num_consumer_types = len(consumer_types_params)
        self.num_actions = len(action_values)
        
        # Store historical data for plotting
        self.history_proportions = []
        self.history_price_distribution = []
        self.history_aggregator_profit = []
        self.history_total_consumption = []
        self.history_voltage_violations = []

        self.prev_total_consumption = PREV_TOTAL_CONSUMPTION_AVG # Initialize for first step

    def run_simulation(self, time_steps, dt):
        """
        Runs the population game simulation over specified time steps.
        Args:
            time_steps (int): Number of simulation steps.
            dt (float): Time step for replicator dynamics.
        """
        current_price_distribution = np.ones(len(self.price_options)) / len(self.price_options) # Initial uniform price distribution

        for t in range(time_steps):
            # Update renewable generation with seasonal pattern and noise
            for params in self.consumer_types_params:
                base_R = params.get('R_base', params['R_i'])
                seasonal = 1 + R_DYNAMIC_AMPLITUDE * np.sin(2 * np.pi * t / time_steps)
                noise = np.random.normal(1.0, R_NOISE_STD)
                params['R_i'] = max(base_R * seasonal * noise, 0.0)

            # --- Aggregator's Turn: Optimize Price Distribution ---
            # Define bounds for probabilities (0 to 1)
            bounds = [(0, 1)] * len(self.price_options)
            # Define constraint: sum of probabilities must be 1
            constraints = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0})

            # Initial guess for optimization (can be current_price_distribution)
            x0 = current_price_distribution

            # Pass current consumer population state to the objective function
            current_consumer_state = {
                'proportions': self.proportions,
                'consumer_types_params': self.consumer_types_params
            }

            # Perform optimization
            result = minimize(
                aggregator_profit_objective,
                x0,
                args=(current_consumer_state, self.aggregator_params, self.price_options, 
                      S_II_AVG, INITIAL_VOLTAGE_PU, self.prev_total_consumption),
                method='SLSQP', # Sequential Least Squares Programming
                bounds=bounds,
                constraints=constraints,
                options={'disp': False, 'ftol': 1e-6} # ftol for convergence tolerance
            )
            
            # Update aggregator's price distribution
            current_price_distribution = result.x
            # Ensure probabilities are non-negative and sum to 1 due to floating point errors
            current_price_distribution[current_price_distribution < 0] = 0
            current_price_distribution /= np.sum(current_price_distribution)

            # --- Consumers' Turn: Update Proportions via Replicator Dynamics ---
            new_proportions = np.copy(self.proportions)
            total_current_consumption = 0.0

            for type_idx in range(self.num_consumer_types):
                consumer_params = self.consumer_types_params[type_idx]
                current_type_proportions = self.proportions[type_idx]
                
                payoffs_for_actions = np.array([
                    perceived_payoff(action_val, current_price_distribution, self.price_options, consumer_params)
                    for action_val in self.action_values
                ])
                
                # Calculate average payoff for this consumer type
                avg_payoff_for_type = np.sum(current_type_proportions * payoffs_for_actions)
                
                # Apply replicator dynamics: x' = x (u - \bar{u}) dt
                # where u is payoff for the action and \bar{u} is the average payoff
                d_proportions = current_type_proportions * (payoffs_for_actions - avg_payoff_for_type) * dt
                new_proportions[type_idx] += d_proportions
                
                # Ensure proportions remain non-negative and sum to 1
                new_proportions[type_idx][new_proportions[type_idx] < 0] = 0
                sum_new_props = np.sum(new_proportions[type_idx])
                if sum_new_props > 0:
                    new_proportions[type_idx] /= sum_new_props
                else: # Avoid division by zero if all proportions become zero (unlikely but for robustness)
                    new_proportions[type_idx] = np.ones(self.num_actions) / self.num_actions # Reset to uniform if all vanish

                # Calculate total consumption for this type
                avg_action = np.sum(new_proportions[type_idx] * self.action_values)
                total_current_consumption += avg_action
                consumer_params['a_hat_i'] = avg_action

            self.proportions = new_proportions

            # --- Record Metrics ---
            self.history_proportions.append(np.copy(self.proportions))
            self.history_price_distribution.append(np.copy(current_price_distribution))
            self.history_total_consumption.append(total_current_consumption)

            # Calculate voltage deviation and violations for recording
            voltage_deviation = calculate_voltage_deviation(
                total_current_consumption, S_II_AVG, INITIAL_VOLTAGE_PU, self.prev_total_consumption
            )
            # A simple heuristic for voltage violations: if deviation is above a threshold
            voltage_violation_threshold = 0.02 # e.g., 2% deviation from 1 p.u.
            num_violations = 1 if voltage_deviation > voltage_violation_threshold else 0
            self.history_voltage_violations.append(num_violations)

            # Recalculate aggregator profit with the *actual* resulting consumption
            # (not the predicted one from optimization)
            agg_profit = -aggregator_profit_objective(
                current_price_distribution, current_consumer_state, self.aggregator_params, 
                self.price_options, S_II_AVG, INITIAL_VOLTAGE_PU, self.prev_total_consumption
            )
            self.history_aggregator_profit.append(agg_profit)

            self.prev_total_consumption = total_current_consumption # Update for next step

    def plot_results(self):
        """Plots the simulation results."""
        time_axis = np.arange(TIME_STEPS) * DT

        # Plot 1: Consumer Action Proportions
        plt.figure(figsize=(15, 10))
        for type_idx in range(self.num_consumer_types):
            plt.subplot(self.num_consumer_types, 1, type_idx + 1)
            type_label = f"Type {chr(65 + type_idx)}" # A, B, C, D
            
            # Extract proportions for this type over time
            type_proportions_history = np.array([h[type_idx] for h in self.history_proportions])
            
            for action_idx, action_val in enumerate(self.action_values):
                plt.plot(time_axis, type_proportions_history[:, action_idx], 
                         label=f'Action {action_val} kWh')
            
            plt.title(f'{type_label} Consumer Action Proportions Over Time')
            plt.xlabel('Time')
            plt.ylabel('Proportion')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
        plt.suptitle('Evolution of Consumer Action Proportions', y=1.02, fontsize=16)
        plt.show()

        # Plot 2: Aggregator Price Distribution
        plt.figure(figsize=(10, 6))
        price_dist_history = np.array(self.history_price_distribution)
        for i, price_val in enumerate(self.price_options):
            plt.plot(time_axis, price_dist_history[:, i], label=f'Price {price_val} unit/kWh')
        plt.title('Aggregator Electricity Price Distribution Over Time')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot 3: Aggregator Profit
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, self.history_aggregator_profit, label='Aggregator Profit')
        plt.title('Aggregator Profit Over Time')
        plt.xlabel('Time')
        plt.ylabel('Profit (units)')
        plt.grid(True)
        plt.show()

        # Plot 4: Total Consumption and Voltage Violations
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, self.history_total_consumption, label='Total Consumption')
        # Overlay voltage violations (as a binary indicator)
        plt.plot(time_axis, np.array(self.history_voltage_violations) * max(self.history_total_consumption)/2, 
                 'r--', label='Voltage Violation (scaled)') # Scale for visibility
        plt.title('Total Consumption and Voltage Violations Over Time')
        plt.xlabel('Time')
        plt.ylabel('Total Consumption (kWh) / Violation Indicator')
        plt.legend()
        plt.grid(True)
        plt.show()


# --- 5. Main Execution ---
if __name__ == "__main__":
    # Define consumer types parameters (beta, gamma, alpha, R_i, v_i, S_ii, a_hat_i)
    # R_i, v_i, S_ii, a_hat_i are simplified to constants for all consumers in this aggregate model
    # In a more detailed model, these would vary per individual consumer or node.
    
    # Consumer types based on Table I from the paper, adapted for population game
    # Type A: Highly comfort-seeking and grid aware, rational
    # Type B: Comfort-seeking and grid aware, slightly less rational
    # Type C: Comfort-seeking and grid unaware, less rational
    # Type D: Low comfort-seeking and grid aware, very irrational
    
    consumer_types_params = [
        {'beta': 0.85, 'gamma': 1.0, 'alpha': 1.0, 'R_i': 10.0, 'R_base': 10.0, 'v_i': INITIAL_VOLTAGE_PU, 'S_ii': S_II_AVG, 'a_hat_i': 5.0}, # Type A
        {'beta': 0.75, 'gamma': 1.0, 'alpha': 0.9, 'R_i': 10.0, 'R_base': 10.0, 'v_i': INITIAL_VOLTAGE_PU, 'S_ii': S_II_AVG, 'a_hat_i': 5.0}, # Type B
        {'beta': 0.75, 'gamma': 0.0, 'alpha': 0.7, 'R_i': 10.0, 'R_base': 10.0, 'v_i': INITIAL_VOLTAGE_PU, 'S_ii': S_II_AVG, 'a_hat_i': 5.0}, # Type C
        {'beta': 0.40, 'gamma': 1.0, 'alpha': 0.5, 'R_i': 10.0, 'R_base': 10.0, 'v_i': INITIAL_VOLTAGE_PU, 'S_ii': S_II_AVG, 'a_hat_i': 5.0}, # Type D
    ]

    num_consumer_types = len(consumer_types_params)
    num_actions = len(ACTION_VALUES)

    # Initial proportions for each action within each consumer type
    # Start with a uniform distribution for all actions for all types
    initial_proportions = np.full((num_consumer_types, num_actions), 1.0 / num_actions)

    # Aggregator parameters dictionary
    aggregator_params = {
        'RHO_DA': RHO_DA,
        'A_AGGREGATOR': A_AGGREGATOR,
        'RHO_RT': RHO_RT,
        'THETA_AGGREGATOR': THETA_AGGREGATOR,
        'RISK_COEFF': AGG_RISK_COEFF
    }

    # Create and run the simulation
    print("Starting population game simulation...")
    sim = PopulationGameSimulation(
        consumer_types_params,
        initial_proportions,
        ACTION_VALUES,
        PRICE_OPTIONS,
        aggregator_params
    )
    sim.run_simulation(TIME_STEPS, DT)
    print("Simulation finished. Plotting results...")

    # Plot results
    sim.plot_results()
