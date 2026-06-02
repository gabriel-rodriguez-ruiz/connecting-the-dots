# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 08:13:00 2026

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Space
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')
import scipy
from pathlib import Path
from two_dimensional_electron_gas import TwoDimensionalElectronGas

c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 #50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.788e-2 # meV/TT


Delta =  0.08 #0.08  #2*0.122 # 0.08 #0.08   #  meVs
mu = E_F  # 623 Delta #50.6  #  meV
Lambda = 15  #15 #187*Delta/2 # meV*nm    # 8 * Delta  #0.644 meV 
theta = np.pi/2
cut_off = 2 * k_F # 1.1 k_F

B = 3*Delta   #0.28*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)

q_B_constant = 0 #0.024/8
phi_x = 0    #q_B_constant * B  #0.0004  #0.024 * 0.5 * Delta
phi_y = 0

k_1 = (-Lambda + np.sqrt(Lambda**2 
                             + 4*gamma*mu)) / (2*gamma)
k_2 = (Lambda + np.sqrt(Lambda**2
                             + 4*gamma*mu)) / (2*gamma)
k_values = [np.linspace(0*k_F, 0.99*k_1, 100, endpoint=False),
            np.append(np.linspace(0.99*k_1, 1.01*k_1, 100, endpoint=False),
                     [np.linspace(1.01*k_1, 0.99*k_2, 100, endpoint=False),
                      np.linspace(0.99*k_2, 1.01*k_2, 100, endpoint=False)]),
            np.linspace(1.01*k_2, cut_off, 100)]

theta_values = np.linspace(0, 2*np.pi, 100)
N = 1

Electron_Gas = TwoDimensionalElectronGas(mu, Delta, B_x, B_y, gamma, Lambda)

n_calls = 10  #15
n_initial_points = 2  #5
# Define the skewed Mexican hat function in 1D
def wrapper_function(x):
    def function(phi_x):
        energy_phi_2DEG = Electron_Gas.get_fundamental_energy(k_values,
                                      theta_values, phi_x, phi_y, N)
        energy_phi_x = (1/2 * energy_phi_2DEG
        +  np.pi/2 * cut_off**2 *
        (2*gamma*(phi_x)**2 - 2*mu + gamma*cut_off**2) )
        return energy_phi_x
    return function(x)

# Expected equilibrium phase
phi_star_1 = -Lambda*B/(4*gamma**2*k_F**2)
phi_star_2 = -B/(2*gamma*k_F)

# Define the search space
if B>Delta:
    search_space = [(1.5*phi_star_2, -1.5*phi_star_2)]
    initial_points = [[phi_star_2], [0.], [-phi_star_2]]
else:
    search_space = [(1.5*phi_star_1, 0)]
    initial_points = [[phi_star_1]]


#%% Plot fundamental energy


phi_x_values = np.linspace(search_space[0][0], search_space[0][1], 10)
energy_phi_x = np.zeros_like(phi_x_values)

for i, phi_x in enumerate(phi_x_values):
    print(phi_x)
    energy_phi_x[i] = wrapper_function(phi_x)
    


# Vectorized version for plotting
x_plot = phi_x_values
y_plot = energy_phi_x

# Plot the function to see what we're dealing with
plt.figure(figsize=(12, 5))
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Skewed Mexican Hat')
plt.xlabel('x')
plt.ylabel('E_0(x)')
# plt.title('1D Skewed Mexican Hat Function')
plt.grid(True, alpha=0.3)
# plt.legend()
plt.show()

#%%

# Bayesian Optimization setup
def objective_function(x):
    """Objective function to minimize"""
    return wrapper_function(x[0])

print("Running Bayesian Optimization...")

# Run Bayesian Optimization with only 25 function evaluations!
result = gp_minimize(
    func=objective_function,
    dimensions=search_space,
    n_calls=n_calls,                    # Only 15 expensive evaluations!
    n_initial_points=n_initial_points,           # Start with 5 random points
    random_state=None,
    acq_func='LCB',  #Lower Confidence Bound (more exploratory) #"EI"  Expected Improvement
    noise=0.0,
    initial_point_generator="lhs",
    x0=initial_points
)

print("\n=== RESULTS ===")
print(f"Global minimum found at: x = {result.x[0]:.6f}")
print(f"Function value at minimum: f(x) = {result.fun:.6f}")

#%%

# Find the true global minimum for comparison (we only know this because we can evaluate densely)
true_min_idx = np.argmin(y_plot)
true_min_x = x_plot[true_min_idx]
true_min_val = y_plot[true_min_idx]

print(f"True global minimum: x = {true_min_x:.6f}, f(x) = {true_min_val:.6f}")
print(f"Error in position: {abs(result.x[0] - true_min_x):.6f}")

# Plot the results
plt.figure(figsize=(15, 5))

# Plot 1: Function with optimization points
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Fundamental energy', alpha=0.7)

# Plot all evaluation points
for i, (x_val, y_val) in enumerate(zip([xi[0] for xi in result.x_iters], result.func_vals)):
    color = 'red' if i < 5 else 'green'  # Initial points in red, BO points in green
    marker = 'o' if i < 5 else 's'
    alpha = 0.7 if i < 5 else 1.0
    plt.scatter(x_val, y_val, c=color, marker=marker, alpha=alpha, s=50)

# Mark the best point found
plt.scatter(result.x[0], result.fun, c='gold', marker='*', s=200, 
           label=f'Best found: x={result.x[0]:.5f}', edgecolors='black')

plt.xlabel(r'$q_x$')
plt.ylabel('f(x)')
plt.title('Bayesian Optimization Progress\n(Red: Initial, Green: BO, Star: Best)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Convergence plot
plt.subplot(1, 2, 2)
plot_convergence(result)
plt.title('Convergence Plot')

plt.tight_layout()
plt.show()

# Print the evaluation history
print("\n=== EVALUATION HISTORY ===")
print("Iteration |      x      |    f(x)    |   Best So Far")
print("-" * 55)
best_so_far = float('inf')
for i, (x_val, f_val) in enumerate(zip(np.array(result.x_iters), result.func_vals)):
    if f_val < best_so_far:
        best_so_far = f_val
    print(f"{i+1:>8} | {x_val[0]:>10.5f} | {f_val:>9.9f} | {best_so_far:>12.9f}")
    
