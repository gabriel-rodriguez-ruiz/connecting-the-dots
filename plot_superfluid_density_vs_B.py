#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:17:50 2026

@author: gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,
})

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_B_in_1.6_(0.0-3.0)_Delta=0.08_lambda=15_points=19_N=100_h=1e-05_T=False_beta=100.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
k_F = Data["k_F"]
# beta = Data["beta"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]

fig, ax = plt.subplots()
# ax.plot(B_values/Delta, superfluid_density_xx, "v", label="perpendicular",
#              color="green")
ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v", label="perpendicular")

# ax.plot(B_values/Delta, superfluid_density_yy, "o", label="parallel",
#             color="red")
ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o", label="parallel")

ax.legend()
ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$D_s$")
ax.set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")