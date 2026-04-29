#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:05:51 2026

@author: gabriel
"""

import numpy as np
from pauli_matrices import (tau_0, tau_z, tau_x, sigma_0, sigma_y, sigma_x,
                            sigma_z)
import scipy
import matplotlib.pyplot as plt

class TwoDimensionalElectronGas():
    """A class for 2DEG energy spectra calculations.

    Parameters
    -----------
    
    mu : float
        Chemical potential.
    Delta : float
        Superconducting gap.
    B_x : float
        Magnetic field in x.
    B_y : float
        Magnetic field in y.
    gamma : float
        Inverse pf mass.
    Lambda : float
        Spin-orbit coupling magnitude.
    """
    def __init__(self, mu, Delta, B_x, B_y, gamma, Lambda):
        self.mu = mu
        self.Delta = Delta
        self.B_x = B_x
        self.B_y = B_y
        self.gamma = gamma
        self.Lambda = Lambda
        self.k_F = np.sqrt(self.mu / self.gamma )
    def get_Hamiltonian_in_polars(self, k, theta, phi_x, phi_y):
        r"""Return the 2DEG Hamiltonian matrix for a given k.
        
        .. math ::
            H = \frac{1}{2} \Biggl\{ \biggl(\gamma \left[ ( k_x
                                                + \phi_x )^2 
                                                + k_y^2 \right]
                - \mu \biggr) \frac{\tau_0 + \tau_z}{2} \sigma_0
                - \biggl(\gamma \left[ ( -k_x + \phi_x )^2 
                                                    + k_y^2 \right]
                    - \mu \biggr) \frac{\tau_0 - \tau_z}{2} \sigma_0\\
                - B_x \tau_0 \sigma_x
                - B_y \tau_0 \sigma_y
                - \Delta \tau_x \sigma_0
                + \lambda (k_x + \phi_x) \frac{\tau_0
                                               + \tau_z}{2}\sigma_0
                + \lambda (-k_x + \phi_x) \frac{\tau_0
                                    - \tau_z}{2} \sigma_0 \\
                - \lambda k_y \tau_z\sigma_x
                \Biggr\}
                
        Parameters
        ----------
        
        k : float
            Radial wave number.
        theta : float
            Polar wave number.
        phi_x : float
            Twisted boundary condition phase in x.
        phi_y : float
            Twisted boundary condition phase in y.
        """
        k_x = k * np.cos(theta)
        k_y = k * np.sin(theta)
        chi_k_plus = (self.gamma * ( (k_x + phi_x)**2 + (k_y + phi_y)**2)
                      - self.mu )
        chi_k_minus = (self.gamma * ( (-k_x + phi_x)**2 + (-k_y + phi_y)**2 )
                       - self.mu )
        return ( chi_k_plus * np.kron( ( tau_0 + tau_z )/2, sigma_0)
                       - chi_k_minus * np.kron( ( tau_0 - tau_z )/2, sigma_0)
                       - self.B_x * np.kron(tau_0, sigma_x)
                       - self.B_y * np.kron(tau_0, sigma_y)
                       - self.Delta * np.kron(tau_x, sigma_0)
                       + self.Lambda * (k_x + phi_x) * np.kron(
                           ( tau_0 + tau_z )/2, sigma_y
                           )
                       + self.Lambda * (-k_x + phi_x) * np.kron(
                           ( tau_0 - tau_z )/2, sigma_y
                           )
                       - self.Lambda * (k_y + phi_y) * np.kron(
                           ( tau_0 + tau_z )/2, sigma_x )
                       - self.Lambda * (-k_y + phi_y) * np.kron(
                           ( tau_0 - tau_z )/2, sigma_x )
                 )
    def get_Eigenvectors_in_polars(self, k_values, theta_values, phi_x, phi_y):
        """Return the eigenvalues and eigenvectors of the Hamiltonian
        for a set of k_values and theta_values.
        
        Parameters
        ----------
        
        k_values : ndarray
            Radial k values.
        theta_values : ndarray
            Polar theta values.        
        phi_x : float
            Twisted boundary condition phase in x.
        phi_y : float
            Twisted boundary condition phase in y.    
       """
        eigenvalues = np.zeros((len(k_values), len(theta_values), 4),
                               dtype=complex)
        eigenvectors = np.zeros((len(k_values), len(theta_values), 4, 4),
                                dtype=complex)
        for i, k in enumerate(k_values):
            for j, theta in enumerate(theta_values):
                H = self.get_Hamiltonian_in_polars(k, theta, phi_x, phi_y)
                eigenvalues[i, j, ], eigenvectors[i, j, :, :] = \
                    scipy.linalg.eigh(H)
        return eigenvalues, eigenvectors
    def get_Energies_in_polars(self, k_values, theta_values, phi_x, phi_y):
        """ Return the energies connecting the dots in the radial
            direction.
       """
        E = np.zeros((len(k_values), len(theta_values), 4), dtype=complex)
        def eigh_interval(k, theta):
            e, psi = self.get_Eigenvectors_in_polars([k], [theta], phi_x,
                                                     phi_y)
            return e[0][0], psi[0][0]
        for i, theta in enumerate(theta_values):
            e, psi =  eigh_interval(k_values[0], theta)
            E[0, i, :] = e
            for j, k in enumerate(k_values[1:]):
                e2, psi2 = eigh_interval(k, theta)
                Q = np.abs(psi.T.conj() @ psi2)  # Overlap matrix
                assignment = scipy.optimize.linear_sum_assignment(-Q)[1]
                E[j+1, i, :] = e2[assignment]
                psi = psi2[:, assignment]
        return np.real(E)
    def plot_energy_bands(self, k_values, theta_values, phi_x, phi_y):
        """ Plot energy bands.
       """
        Energies = self.get_Energies_in_polars(k_values, theta_values,
                                               phi_x, phi_y)
        fig, ax = plt.subplots()
        for i, theta in enumerate(theta_values):
            ax.plot(k_values/self.k_F, Energies[:, i, :], "o-", markersize=3,
                    label=r"$\theta/\pi=$"+f"{np.round(theta/np.pi, 3)}")
        ax.set_xlabel(r"$k/k_F$")
        ax.set_ylabel(r"$E(k)$")
        ax.legend()
        plt.grid()
        return fig, ax
    def find_radial_energy_roots(self, Energies, k_values):
        """ Search radial energy roots from energy values.
       """
        roots = []
        for i in range(4):
            f_values = Energies[:, i]
            sign_changes = np.where(np.diff(np.sign(f_values)))[0]
            brackets = [(k_values[j], k_values[j+1]) for j in sign_changes]
            for bracket in brackets:
                f_scalar = scipy.interpolate.CubicSpline(k_values, f_values)
                sol = scipy.optimize.root_scalar(f_scalar, bracket=bracket,
                                                 method='brentq')
                if sol.converged:
                    roots.append(sol.root)
        return np.sort(np.unique(np.round(roots, decimals=8)))
    def get_interpolation_of_energy(self, Energies, k_values, theta_value,
                                    phi_x, phi_y, N):
        """ Interpolation of energy.
        """
        def get_swaps(start, target):
            arr = list(start)
            target = list(target)
            swaps = 0
            for i in range(len(arr)):
                if arr[i] != target[i]:
                    # Encontrar dónde está el elemento que debería ir aquí
                    target_idx = arr.index(target[i])
                    # Intercambiar
                    arr[i], arr[target_idx] = arr[target_idx], arr[i]
                    swaps += 1
            return swaps
        root_index = []
        N_index = 5
        extended_Energies = np.zeros((0, 4))
        extended_k_values = np.zeros(0)
        roots = self.find_radial_energy_roots(Energies, k_values)
        k_1 = (-self.Lambda + np.sqrt(self.Lambda**2 
                                     + 4*self.gamma*self.mu)) / (2*self.gamma)
        k_2 = (self.Lambda + np.sqrt(self.Lambda**2
                                     + 4*self.gamma*self.mu)) / (2*self.gamma)
        roots = np.sort(np.append(roots, np.array([k_1, k_2])))
        for i in range(len(roots)):
            root_index.append(np.max(np.where(
                k_values<roots[i]))-N_index)
            root_index.append(np.min(np.where(
                k_values>roots[i]))+N_index)
        root_index = np.unique(root_index)
        extended_Energies = np.concatenate((extended_Energies,
                               Energies[:root_index[0], :]))
        extended_k_values = np.concatenate((extended_k_values,
                                            k_values[:root_index[0]]))
        for i in range(len(root_index)-1):
            #Indeces that would sort the energy before concatenation
            indeces = np.argsort(Energies[root_index[i]-1, :])
            indeces = np.argsort(indeces)
            extended_Energies = np.concatenate((extended_Energies,
                               self.get_Energies_in_polars(
                               np.linspace(k_values[root_index[i]],
                                           k_values[root_index[i+1]],
                                           N),
                               [theta_value], phi_x, phi_y)[:,0,
                                                    indeces]),
                                               axis=0)
            extended_k_values = np.concatenate((extended_k_values,
                                    np.linspace(k_values[root_index[i]],
                                        k_values[root_index[i+1]], N)
                                    ))
        extended_Energies = np.concatenate((extended_Energies,
                                            Energies[root_index[-1]:,:]))
        extended_k_values = np.concatenate((extended_k_values,
                                            k_values[root_index[-1]:]))
        return extended_Energies, extended_k_values, roots
    def find_pockets(self, k_values, theta_values, phi_x, phi_y):
        """Return the position of pockets.
        """
        Energies = np.zeros((len(k_values), len(theta_values), 4))
        roots = []
        Energies = self.get_Energies_in_polars(k_values,
                                               theta_values, phi_x, phi_y)
        for i, theta in enumerate(theta_values):
            roots.append(self.find_radial_energy_roots(Energies[:,i,:],
                                                        k_values))
        return roots
    def get_radial_integral(self, k_values, theta_value, phi_x, phi_y, N):
        """Integration in k.
        """
        low_radius_values, radius_values_k_F, high_radius_values = k_values
        low_integral = np.zeros(2)
        for i in range(2):
            f = lambda r: ( r * self.get_Energies_in_polars([r],
                    [theta_value], phi_x, phi_y)[0][0][i] )
            low_integral[i], abserr = scipy.integrate.quad(f, 0,
                                      np.max(low_radius_values))
        Energies_k_F = self.get_Energies_in_polars(radius_values_k_F,
                                                   [theta_value],
                                                   phi_x, phi_y)
        extended_Energies, extended_k_values, roots = self.\
                        get_interpolation_of_energy(Energies_k_F[:, 0, :],
                                                    radius_values_k_F, 
                                                    theta_value,
                                        phi_x, phi_y, N)
        integral = np.zeros(4)
        for i in range(4):
            # f = lambda r: ( r * (-1/2) * np. abs(self.get_Energies_in_polars([r],
            #         [theta_value], phi_x, phi_y)[0][0][i] ) )
            # integral[i], abserr = scipy.integrate.quad(f,
            #                           np.min(extended_k_values),
            #                           np.max(extended_k_values),
            #                           points=roots)
            integral[i] = scipy.integrate.trapezoid(
                extended_k_values * (-1/2) * np.abs(extended_Energies[:, i]),
                extended_k_values, axis=0,
                dx=np.diff(extended_k_values)[0])
        high_integral = np.zeros(2)
        for i in range(2):
            f = lambda r: ( r * self.get_Energies_in_polars([r],
                    [theta_value], phi_x, phi_y)[0][0][i] )
            high_integral[i], abserr = scipy.integrate.quad(f,
                                      np.max(extended_k_values),
                                      np.max(high_radius_values))
        return low_integral, integral, high_integral
    def get_radial_integral_at_finite_T_in_x(self, k_values, theta_value,
                                        phi_x, phi_y, N, h, beta):
        """Integration in k.
        """
        low_radius_values, radius_values_k_F, high_radius_values = k_values
        low_integral = np.zeros(2)
        for i in range(2):
            f = lambda r: ( r * (self.get_Energies_in_polars([r],
                    [theta_value], phi_x+h, phi_y)[0][0][i] 
                                 - self.get_Energies_in_polars([r],
                                     [theta_value], phi_x-h, phi_y)[0][0][i])
                           /(2*h))
            low_integral[i], abserr = scipy.integrate.quad(f, 0,
                                      np.max(low_radius_values))
        Energies_k_F = self.get_Energies_in_polars(radius_values_k_F,
                                                   [theta_value],
                                                   phi_x, phi_y)
        extended_Energies, extended_k_values, roots = self.\
                        get_interpolation_of_energy(Energies_k_F[:, 0, :],
                                                    radius_values_k_F, 
                                                    theta_value,
                                        phi_x, phi_y, N)
        extended_Energies_positive = self.get_Energies_in_polars(
            extended_k_values,
            [theta_value],
            phi_x + h, phi_y)
        extended_Energies_negative =  self.get_Energies_in_polars(
            extended_k_values,
            [theta_value],
            phi_x - h, phi_y)
        integral = np.zeros(4)
        for i in range(4):
            integral[i] = scipy.integrate.trapezoid(
                extended_k_values * (extended_Energies_positive[:, 0, i]-
                                     extended_Energies_negative[:, 0, i])/(2*h)
                * self.Fermi_function(extended_Energies[:, i], beta)
                ,
                extended_k_values, axis=0,
                dx=np.diff(extended_k_values)[0])
        high_integral = np.zeros(2)
        for i in range(2):
            f = lambda r: ( r * (self.get_Energies_in_polars([r],
                    [theta_value], phi_x+h, phi_y)[0][0][i] 
                                 - self.get_Energies_in_polars([r],
                                     [theta_value], phi_x-h, phi_y)[0][0][i])
                           /(2*h))
            high_integral[i], abserr = scipy.integrate.quad(f,
                                      np.max(extended_k_values),
                                      np.max(high_radius_values))
        return low_integral, integral, high_integral
    def get_radial_integral_at_finite_T_in_y(self, k_values, theta_value,
                                        phi_x, phi_y, N, h, beta):
        """Integration in k.
        """
        low_radius_values, radius_values_k_F, high_radius_values = k_values
        low_integral = np.zeros(2)
        for i in range(2):
            f = lambda r: ( r * (self.get_Energies_in_polars([r],
                    [theta_value], phi_x, phi_y+h)[0][0][i] 
                                 - self.get_Energies_in_polars([r],
                                     [theta_value], phi_x, phi_y-h)[0][0][i])
                           /(2*h))
            low_integral[i], abserr = scipy.integrate.quad(f, 0,
                                      np.max(low_radius_values))
        Energies_k_F = self.get_Energies_in_polars(radius_values_k_F,
                                                   [theta_value],
                                                   phi_x, phi_y)
        extended_Energies, extended_k_values, roots = self.\
                        get_interpolation_of_energy(Energies_k_F[:, 0, :],
                                                    radius_values_k_F, 
                                                    theta_value,
                                        phi_x, phi_y, N)
        extended_Energies_positive = self.get_Energies_in_polars(
            extended_k_values,
            [theta_value],
            phi_x, phi_y + h)
        extended_Energies_negative =  self.get_Energies_in_polars(
            extended_k_values,
            [theta_value],
            phi_x, phi_y - h)
        integral = np.zeros(4)
        for i in range(4):
            integral[i] = scipy.integrate.trapezoid(
                extended_k_values * (extended_Energies_positive[:, 0, i]-
                                     extended_Energies_negative[:, 0, i])/(2*h)
                * self.Fermi_function(extended_Energies[:, i], beta)
                ,
                extended_k_values, axis=0,
                dx=np.diff(extended_k_values)[0])
        high_integral = np.zeros(2)
        for i in range(2):
            f = lambda r: ( r * (self.get_Energies_in_polars([r],
                    [theta_value], phi_x, phi_y+h)[0][0][i] 
                                 - self.get_Energies_in_polars([r],
                                     [theta_value], phi_x, phi_y-h)[0][0][i])
                           /(2*h))
            high_integral[i], abserr = scipy.integrate.quad(f,
                                      np.max(extended_k_values),
                                      np.max(high_radius_values))
        return low_integral, integral, high_integral
    def get_fundamental_energy(self, k_values, theta_values, phi_x, phi_y,
                               N):
        radial_integral = np.zeros_like(theta_values)
        for i, theta in enumerate(theta_values):
            low_integral, integral, high_integral = self.get_radial_integral(
            k_values, theta, phi_x, phi_y, N)
            radial_integral[i] = (np.sum(low_integral) + np.sum(integral)
                                  + np.sum(high_integral))
        fundamental_energy = scipy.integrate.trapezoid(
            radial_integral,
            theta_values, axis=0,
            dx=np.diff(theta_values)[0])
        return fundamental_energy
    def Fermi_function(self, energy, beta):
        return 1 / (1 + np.exp(beta * energy))
    def get_current_in_x(self, k_values, theta_values, phi_x, phi_y, N, h,
                         T, beta):
        if T==False:
            fundamental_energy = np.zeros(2)
            fundamental_energy[0] = self.get_fundamental_energy(k_values,
                                     theta_values, phi_x + h, phi_y, N)
            fundamental_energy[1] = self.get_fundamental_energy(k_values,
                                     theta_values, phi_x - h, phi_y, N)   
            current = 1/2*(fundamental_energy[0] - fundamental_energy[1]
                           )/ (2*h)
        else:
            radial_integral = np.zeros_like(theta_values)
            for i, theta in enumerate(theta_values):
                low_integral, integral, high_integral = self.\
                    get_radial_integral_at_finite_T_in_x(
                        k_values, theta, phi_x, phi_y, N, h, beta)
                radial_integral[i] = (np.sum(low_integral) + np.sum(integral)
                                      + np.sum(high_integral))
            current = 1/2 * scipy.integrate.trapezoid(
                radial_integral,
                theta_values, axis=0,
                dx=np.diff(theta_values)[0])
        return current
    def get_current_in_y(self, k_values, theta_values, phi_x, phi_y, N, h,
                         T, beta):
        if T==False:
            fundamental_energy = np.zeros(2)
            fundamental_energy[0] = self.get_fundamental_energy(k_values,
                                     theta_values, phi_x, phi_y + h, N)
            fundamental_energy[1] = self.get_fundamental_energy(k_values,
                                     theta_values, phi_x, phi_y - h, N)   
            current = 1/2*(fundamental_energy[0] - fundamental_energy[1]
                           )/ (2*h)
        else:
            radial_integral = np.zeros_like(theta_values)
            for i, theta in enumerate(theta_values):
                low_integral, integral, high_integral = self.\
                    get_radial_integral_at_finite_T_in_y(
                k_values, theta, phi_x, phi_y, N, h, beta)
                radial_integral[i] = (np.sum(low_integral) + np.sum(integral)
                                      + np.sum(high_integral))
            current = 1/2 * scipy.integrate.trapezoid(
                radial_integral,
                theta_values, axis=0,
                dx=np.diff(theta_values)[0])
        return current
    def get_density_radial_integral(self, k_values, theta_value, phi_x,
                                    phi_y, N, T, beta):
        """Integration in k.
        """
        low_radius_values, radius_values_k_F, high_radius_values = k_values
        low_integral = np.zeros(2)
        for i in range(2):
            f = lambda r: ( r * 1 )
            low_integral[i], abserr = scipy.integrate.quad(f, 0,
                                      np.max(low_radius_values))
        Energies_k_F = self.get_Energies_in_polars(radius_values_k_F,
                                                   [theta_value],
                                                   phi_x, phi_y)
        extended_Energies, extended_k_values, roots = self.\
                        get_interpolation_of_energy(Energies_k_F[:, 0, :],
                                                    radius_values_k_F, 
                                                    theta_value,
                                        phi_x, phi_y, N)
        integral = np.zeros(4)
        for i in range(4):
            if T==True:
                integral[i] = scipy.integrate.trapezoid(
                    extended_k_values * (1/2)
                    * self.Fermi_function(extended_Energies[:, i], beta),
                    extended_k_values, axis=0,
                    dx=np.diff(extended_k_values)[0])
            else:
                integral[i] = scipy.integrate.trapezoid(
                    extended_k_values * (1/2)
                    * np.heaviside(extended_Energies[:, i], 1),
                    extended_k_values, axis=0,
                    dx=np.diff(extended_k_values)[0])
        high_integral = np.zeros(2)
        for i in range(2):
            f = lambda r: ( r * 1 )
            high_integral[i], abserr = scipy.integrate.quad(f,
                                      np.max(extended_k_values),
                                      np.max(high_radius_values))
        return low_integral, integral, high_integral
    def get_superfluid_density(self, k_values, theta_values,
                               phi_x, phi_y, N, h, cut_off, T, beta):
        phi_x_values = np.array([-h, h])
        phi_y_values = np.array([-h, h])
        current_phi_x = np.zeros_like(phi_x_values)
        current_phi_y = np.zeros_like(phi_y_values)
        for i, phi in enumerate(phi_x_values):
            current_phi_x[i] = self.get_current_in_x(k_values, theta_values,
                                            phi_x + phi, phi_y, N, h, T, beta)
        total_current = current_phi_x + 2*np.pi * cut_off**2 * self.gamma*(
                phi_x + phi_x_values)
        superfluid_density_xx = (total_current[1] - total_current[0])/(2*h)
        for i, phi in enumerate(phi_y_values):
            current_phi_y[i] = self.get_current_in_y(k_values, theta_values,
                                            phi_x, phi_y + phi, N, h, T, beta)
        total_current = current_phi_y + 2*np.pi * cut_off**2 * self.gamma*(
                phi_y + phi_y_values)
        superfluid_density_yy = (total_current[1] - total_current[0])/(2*h)
        return superfluid_density_xx, superfluid_density_yy
    def get_density(self, k_values, theta_values,
                               phi_x, phi_y, N, h, T, beta):
        radial_integral = np.zeros_like(theta_values)
        for i, theta in enumerate(theta_values):
            low_integral, integral, high_integral = self.get_density_radial_integral(
            k_values, theta, phi_x, phi_y, N, T, beta)
            radial_integral[i] = (np.sum(low_integral) + np.sum(integral)
                                  + np.sum(high_integral))
        density = scipy.integrate.trapezoid(
            radial_integral,
            theta_values, axis=0,
            dx=np.diff(theta_values)[0])
        return density