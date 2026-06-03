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
from skopt import gp_minimize

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

k_1 = (-Lambda + np.sqrt(Lambda**2 
                             + 4*gamma*mu)) / (2*gamma)
k_2 = (Lambda + np.sqrt(Lambda**2
                             + 4*gamma*mu)) / (2*gamma)
M = 100
k_values = [np.linspace(0*k_F, 0.99*k_1, M, endpoint=False), # this M value is irrelevant
            np.append(np.linspace(0.99*k_1, 1.01*k_1, M, endpoint=False),
                     [np.linspace(1.01*k_1, 0.99*k_2, M, endpoint=False),
                      np.linspace(0.99*k_2, 1.01*k_2, M, endpoint=False)]),
            np.linspace(1.01*k_2, cut_off, M)]  # this M value is irrelevant

theta_values = np.linspace(0, 2*np.pi, 200)
N = 100
n_cores = 15
points = 1 * n_cores

T = True
beta = 100

minimization = False
n_calls = 10  #15
n_initial_points = 2  #5


parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta, "h": h,
              "Lambda": Lambda, "N": N,
              "cut_off": cut_off
              }

def function(phi_x, Electron_Gas):
    if T==False:
        energy_phi_2DEG = Electron_Gas.get_fundamental_energy(k_values,
                                      theta_values, phi_x, phi_y, N)
    else:
        energy_phi_2DEG = Electron_Gas.get_grand_potential(k_values,
                                      theta_values, phi_x, phi_y, N, beta)
    fundamental_energy_2DEG = (1/2 * energy_phi_2DEG
    +  np.pi/2 * cut_off**2 * (2*gamma*(phi_x)**2 - 2*mu + gamma*cut_off**2) )
    return fundamental_energy_2DEG

def get_minima(search_space, initial_points, Electron_Gas):
    def objective_function(x):
        """Objective function to minimize"""
        return function(x[0], Electron_Gas)
    result = gp_minimize(
        func=objective_function,
        dimensions=search_space,
        n_calls=n_calls,                    # Only 25 expensive evaluations!
        n_initial_points=n_initial_points,           # Start with 10 random points
        random_state=None,
        acq_func='LCB',  #Lower Confidence Bound (more exploratory) #"EI"  Expected Improvement
        noise=0.0,
        initial_point_generator="lhs",
        x0=initial_points
    )
    return result.x[0]

def integrate_B(B):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
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
    
    Electron_Gas = TwoDimensionalElectronGas(mu, Delta, B_x, B_y, gamma,
                                             Lambda)
    if minimization==True:
        phi_x = get_minima(search_space, initial_points, Electron_Gas)   #-np.heaviside(B-Delta, 1)*B/(2*gamma*k_F) - np.heaviside(Delta-B, 0)*Lambda*B/(4*gamma**2*k_F**2)
    else:
        phi_x = 0
    print(phi_x)
    phi_y = 0

    superfluid_density_xx, superfluid_density_yy = Electron_Gas.\
                            get_superfluid_density(k_values, theta_values,
                           phi_x, phi_y, N, h, cut_off, T, beta)
    density = Electron_Gas.get_density(k_values, theta_values,
                               phi_x, phi_y, N, h, T, beta)
    return superfluid_density_xx, superfluid_density_yy, density, phi_x

if __name__ == "__main__":
    B_values = np.linspace(0.1*Delta, 3*Delta, points)
    integrate = integrate_B
    B_direction = f"{theta:.2}"
    # integrate = integrate_B_y
    with multiprocessing.Pool(n_cores) as pool:
        superfluid_density_xx, superfluid_density_yy, density, phi_x = zip(*pool.map(integrate, B_values))
    superfluid_density_xx = np.array(superfluid_density_xx)
    superfluid_density_yy = np.array(superfluid_density_yy)
    density = np.array(density)
    phi_x = np.array(phi_x)
    data_folder = Path("Data/")
    name = f"superfluid_density_B_in_{B_direction}_({np.round(np.min(B_values/Delta),3)}-{np.round(np.max(B_values/Delta),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N={N}_h={h}_T={T}_beta={beta}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open,
             superfluid_density_xx=superfluid_density_xx,
             superfluid_density_yy=superfluid_density_yy,
             density = density,
             phi_x=phi_x,
             B_values=B_values, **parameters)
    print("\007")
