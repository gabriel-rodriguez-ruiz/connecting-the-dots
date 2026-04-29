#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 08:42:11 2026

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from two_dimensional_electron_gas import TwoDimensionalElectronGas
import scipy
import multiprocessing
from pathlib import Path

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
Lambda = 22.5  #15 #187*Delta/2 # meV*nm    # 8 * Delta  #0.644 meV 
theta = np.pi/2
cut_off = 2 * k_F # 1.1 k_F

B = 2*Delta   #0.28*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)

q_B_constant = 0 #0.024/8
phi_x = 0    #q_B_constant * B  #0.0004  #0.024 * 0.5 * Delta
phi_y = 0

k_values = [np.linspace(0*k_F, 0.9*k_F, 100),
            np.linspace(0.9*k_F, 1.1*k_F, 300),
            np.linspace(1.1*k_F, cut_off, 100)]
theta_values = np.linspace(0, 2*np.pi, 200)
N = 100
n_cores = 19
points = 1 * n_cores

Electron_Gas = TwoDimensionalElectronGas(mu, Delta, B_x, B_y, gamma, Lambda)

def integrate_phi_x(phi_x):
    energy_phi_2DEG = Electron_Gas.get_fundamental_energy(k_values,
                                  theta_values, phi_x, phi_y, N)
    fundamental_energy_2DEG = (energy_phi_2DEG
    +  np.pi/2 * cut_off**2 * (2*gamma*(phi_x)**2 - 2*mu + gamma*cut_off**2) )
    return fundamental_energy_2DEG

#%%

if __name__ == "__main__":
    phi_x_values = np.linspace(-1e-4, 1e-4, points)
    integrate = integrate_phi_x   # integrate_phi_x
    with multiprocessing.Pool(n_cores) as pool:
        fundamental_energy_2DEG = pool.map(integrate, phi_x_values)
    fundamental_energy_2DEG = np.array(fundamental_energy_2DEG)
    data_folder = Path("Data/")
    name = f"total_fundamental_energy_B={B}_phi_x_in_({np.round(np.min(phi_x_values), 3)}-{np.round(np.max(phi_x_values/k_F),3)})_Delta={Electron_Gas.Delta}_lambda={np.round(Lambda, 2)}_points={points}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open,
             fundamental_energy_2DEG=fundamental_energy_2DEG,
             phi_x_values=phi_x_values)
    print("\007")
    