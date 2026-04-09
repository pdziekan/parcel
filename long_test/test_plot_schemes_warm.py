"""
This test runs the parcel model (with T > 0 C) using three different microphysics schemes:
- lgrngn (Lagrangian particle-based)
- blk_1m (bulk warm)
- blk_1m_ice (bulk ice)

It plots the evolution of rv, th_d, and total condensed water in different schemes. 
"""

import sys, os
sys.path.insert(0, "../")
sys.path.insert(0, "./")

import numpy as np
from parcel import parcel
from scipy.io import netcdf
import matplotlib.pyplot as plt
from libcloudphxx import common

def run_scheme(scheme, ice_switch, outfile):
    args = dict(
        dt=0.1,
        z_max=800,
        w=1.0,
        T_0=300,
        r_0=0.022,
        outfile=outfile,
        outfreq=50,
        scheme=scheme,
        ice_switch=ice_switch,
        out_bin='{"radius": {"rght": 1, "moms": [3], "drwt": "wet", "nbin": 1, "lnli": "lin", "left": 1e-15}}'
    )
    parcel(**args)
    with netcdf.netcdf_file(outfile, 'r') as f:
        rv = np.array(f.variables['r_v'][:])
        th_d = np.array(f.variables['th_d'][:])
        z = np.array(f.variables['z'][:])
        if scheme.startswith("blk"):
            r_tot = np.array(f.variables['rc'][:]) + np.array(f.variables['rr'][:])
        else: 
            r_tot =  np.array(f.variables['radius_m3'][:]) *4/3 * np.pi * common.rho_w
    return rv, th_d, r_tot, z

def test_plot_schemes():
    schemes = ["lgrngn", "blk_1m", "blk_1m_ice"]
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1,3, figsize=(12, 6))
    for scheme in schemes:
        if scheme == "blk_1m_ice":
            rv, th_d, r_tot, z = run_scheme("blk_1m", True, f"test_{scheme}.nc")
        else:
            rv, th_d, r_tot, z = run_scheme(scheme, False, f"test_{scheme}.nc")
        ax[0].plot(rv, z, label=f"{scheme}", linestyle ='--' if scheme=='blk_1m_ice' else '-')
        ax[1].plot(th_d, z, label=f"{scheme}", linestyle ='--' if scheme=='blk_1m_ice' else '-')
        ax[2].plot(r_tot, z, label=f"{scheme}", linestyle ='--' if scheme=='blk_1m_ice' else '-')
        os.remove(f"test_{scheme}.nc")
        
    ax[0].set_ylabel("Height [m]")
    ax[0].set_xlabel("Water vapor mixing ratio")
    ax[1].set_xlabel("Dry potential temperature")
    ax[2].set_xlabel("Total condensed water mixing ratio")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.tight_layout()
    plt.savefig("plots/outputs/plot_schemes_warm.svg")
    