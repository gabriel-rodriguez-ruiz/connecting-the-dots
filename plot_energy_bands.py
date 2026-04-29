#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:38:21 2026

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from two_dimensional_electron_gas import TwoDimensionalElectronGas
import scipy

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

B = 3*Delta   #0.28*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)

q_B_constant = 0 #0.024/8
phi_x = 1e-5 #q_B_constant * B  #0.0004  #0.024 * 0.5 * Delta
phi_y = 0
N = 100
k_values = np.linspace(0.9*k_F, 1.1*k_F, 200)
theta_values =  np.array([1])

Electron_Gas = TwoDimensionalElectronGas(mu, Delta, B_x, B_y, gamma,
                                         Lambda)

Energies = Electron_Gas.get_Energies_in_polars(k_values, theta_values,
                                               phi_x, phi_y)
roots = Electron_Gas.find_radial_energy_roots(Energies[:, 0, :], k_values)

k_1 = (-Lambda + np.sqrt(Lambda**2 
                             + 4*gamma*mu)) / (2*gamma)
k_2 = (Lambda + np.sqrt(Lambda**2
                             + 4*gamma*mu)) / (2*gamma)
roots = np.sort(np.append(roots, np.array([k_1, k_2])))
N_index = 5
root_index = []

for i in range(len(roots)):
    root_index.append(np.max(np.where(
        k_values<roots[i]))-N_index)
    root_index.append(np.min(np.where(
        k_values>roots[i]))+N_index)
root_index = np.unique(root_index)

fig, ax = plt.subplots()
# fig, ax = Electron_Gas.plot_energy_bands(k_values, theta_values, phi_x, phi_y)

# for i, root in enumerate(roots):
#     if i%2==0:
#         ax.plot(np.linspace(0.999*roots[i], 1.001*roots[i+1], 100)/k_F,
#                 Electron_Gas.get_Energies_in_polars(
#                               np.linspace(0.999*roots[i],
#                                           1.001*roots[i+1], 100),
#                                           [0], phi_x, phi_y)[:,0])

for i,theta in enumerate(theta_values):
    extended_Energies, extended_k_values, extended_roots = Electron_Gas.\
                    get_interpolation_of_energy(Energies[:, i, :], k_values, 
                                                theta_values[i],
                                    phi_x, phi_y, N)
    ax.scatter(roots/Electron_Gas.k_F, np.zeros_like(roots))
    # ax.plot(extended_k_values/Electron_Gas.k_F, np.abs(extended_Energies))
    ax.plot(extended_k_values/Electron_Gas.k_F, extended_Energies)

k_1 = (-Lambda + np.sqrt(Lambda**2 + 4*gamma*mu)) / (2*gamma)
k_2 = (Lambda + np.sqrt(Lambda**2 + 4*gamma*mu)) / (2*gamma)

ax.scatter([k_1/k_F, k_2/k_F], [0, 0])
for i in range(4):
    ax.plot(k_values/k_F, Energies[:, 0, i], "o")
    
ax.scatter(k_values[root_index]/k_F, np.zeros_like(root_index), marker="*")
plt.grid()