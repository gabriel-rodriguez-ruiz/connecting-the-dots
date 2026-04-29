#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:39:54 2026

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data_folder = Path(r"./Data")

file_to_open = data_folder / "total_fundamental_energy_B=0.16_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=22.5_points=19.npz"

Data = np.load(file_to_open)
# fundamental_energy = Data["fundamental_energy"]
fundamental_energy_2DEG = Data["fundamental_energy_2DEG"]
phi_x_values = Data["phi_x_values"]

fig, ax = plt.subplots()
ax.plot(phi_x_values, fundamental_energy_2DEG)
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$E$")
# ax.plot(phi_x_values, np.min(fundamental_energy_2DEG)
#         + 2*np.pi*50.6*phi_x_values**2)