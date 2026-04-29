#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:35:14 2026

@author: gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
import scipy
from two_dimensional_electron_gas import TwoDimensionalElectronGas


c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.788e-2 # meV/T

Delta = 0.08   #  meV
mu = 50.6   # 623 Delta #50.6  #  meV
Lambda = 15 # meV*nm    # 8 * Delta  #0.644 meV 

h = 1e-5
phi_x = 0
phi_y = 0
cut_off = 2*k_F # 1.1 k_F

theta = np.pi/2 #np.pi/2   # float

k_values = [np.linspace(0*k_F, 0.9*k_F, 100),
            np.linspace(0.9*k_F, 1.1*k_F, 200),
            np.linspace(1.1*k_F, cut_off, 100)]
theta_values = np.linspace(0, 2*np.pi, 200)
N = 100
n_cores = 19
points = 1 * n_cores

T = False
beta = 100

parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta, "h": h,
              "Lambda": Lambda, "N": N,
              "cut_off": cut_off
              }

def integrate_B(B):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    Electron_Gas = TwoDimensionalElectronGas(mu, Delta, B_x, B_y, gamma,
                                             Lambda)
    superfluid_density_xx, superfluid_density_yy = Electron_Gas.\
                            get_superfluid_density(k_values, theta_values,
                           phi_x, phi_y, N, h, cut_off, T, beta)
    density = Electron_Gas.get_density(k_values, theta_values,
                               phi_x, phi_y, N, h, T, beta)
    return superfluid_density_xx, superfluid_density_yy, density

if __name__ == "__main__":
    B_values = np.linspace(0.*Delta, 3*Delta, points)
    integrate = integrate_B
    B_direction = f"{theta:.2}"
    # integrate = integrate_B_y
    with multiprocessing.Pool(n_cores) as pool:
        superfluid_density_xx, superfluid_density_yy, density = zip(*pool.map(integrate, B_values))
    superfluid_density_xx = np.array(superfluid_density_xx)
    superfluid_density_yy = np.array(superfluid_density_yy)
    density = np.array(density)
    data_folder = Path("Data/")
    name = f"superfluid_density_B_in_{B_direction}_({np.round(np.min(B_values/Delta),3)}-{np.round(np.max(B_values/Delta),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N={N}_h={h}_T={T}_beta={beta}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open,
             superfluid_density_xx=superfluid_density_xx,
             superfluid_density_yy=superfluid_density_yy,
             density = density,
             B_values=B_values, **parameters)
    print("\007")
