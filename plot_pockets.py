#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:50:19 2026

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


Delta =  0.08 #0.08 #0.08  #2*0.122 # 0.08 #0.08   #  meVs
mu = E_F  # 623 Delta #50.6  #  meV
Lambda = 15  #15 #187*Delta/2 # meV*nm    # 8 * Delta  #0.644 meV 
theta = 0   #np.pi/2

B = 3*Delta       #1.1*Delta   #0.28*Delta
B_x = B * np.cos(theta)
B_y = B/2 * np.sin(theta)

q_B_constant = 0 #0.024/8
phi_x = 0 #q_B_constant * B  #0.0004  #0.024 * 0.5 * Delta
phi_y = 0

k_values = np.linspace(0.9*k_F, 1.1*k_F, 300)
theta_values = np.linspace(0, 2*np.pi, 200)

Electron_Gas = TwoDimensionalElectronGas(mu, Delta, B_x, B_y, gamma,
                                         Lambda)

Energies = Electron_Gas.get_Energies_in_polars(k_values, theta_values,
                                               phi_x, phi_y)

pockets = Electron_Gas.find_pockets(k_values, theta_values,
                                             phi_x, phi_y)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
plot_data = [x if len(x) > 0 else np.nan for x in pockets]

ax.scatter(theta_values, [i[0]/Electron_Gas.k_F 
                         if type(i)==np.ndarray
                          else np.nan for i in plot_data])
ax.scatter(theta_values, [i[1]/Electron_Gas.k_F
                          if (type(i)==np.ndarray and len(i)>1) else np.nan
                               for i in plot_data
                               ])
ax.scatter(theta_values, [i[2]/Electron_Gas.k_F
                          if (type(i)==np.ndarray and len(i)>2) else np.nan
                               for i in plot_data
                               ])
ax.scatter(theta_values, [i[3]/Electron_Gas.k_F 
                         if (type(i)==np.ndarray and len(i)>3) else np.nan
                               for i in plot_data
                               ])

ax.set_ylim((0.9, 1.1))

k_1 = (-Lambda + np.sqrt(Lambda**2 + 4*gamma*mu)) / (2*gamma)
k_2 = (Lambda + np.sqrt(Lambda**2 + 4*gamma*mu)) / (2*gamma)

ax.plot(theta_values, (-2*np.cos(theta_values)*phi_x
                       +np.sqrt(4*np.cos(theta_values)**2*phi_x**2
                       -4*(phi_x**2-k_1**2)))/2/Electron_Gas.k_F, "k")

ax.plot(theta_values, (2*np.cos(theta_values)*phi_x
                       +np.sqrt(4*np.cos(theta_values)**2*phi_x**2
                       -4*(phi_x**2-k_1**2)))/2/Electron_Gas.k_F, "k")

ax.plot(theta_values, (-2*np.cos(theta_values)*phi_x
                       +np.sqrt(4*np.cos(theta_values)**2*phi_x**2
                       -4*(phi_x**2-k_2**2)))/2/Electron_Gas.k_F, "k")

ax.plot(theta_values, (2*np.cos(theta_values)*phi_x
                       +np.sqrt(4*np.cos(theta_values)**2*phi_x**2
                       -4*(phi_x**2-k_2**2)))/2/Electron_Gas.k_F, "k")
