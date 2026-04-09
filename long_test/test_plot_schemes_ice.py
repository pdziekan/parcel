"""
This test runs the parcel model with T < 0 C, using three different ice microphysics schemes:
- lgrngn with singular freezing (as in Shima et al., 2020)
- lgrngn with time-dependent freezing (as in Arabas et al., 2025)
- 1-moment bulk (Grabowski, 1999)

It plots the evolution of ice, liquid and vapor mixing ratios for different schemes. 
"""

import sys, os, subprocess
sys.path.insert(0, "../../")
sys.path.insert(0, "./")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from parcel import parcel
from scipy.io import netcdf
from libcloudphxx import common


def run_scheme(scheme, time_dep, outfile):
    args = dict(dt=1., w=1., sd_conc=100,
            z_max = 6000.0,
            T_0 = 265.0,
            RH_0 = 1.,
            scheme = scheme,
            ice_switch=True,
            depo = True,
            ice_nucl=True,
            time_dep_ice_nucl=time_dep,
            aerosol = f'{{"ammonium_sulfate": {{"kappa": 0.61, "rd_insol": 0.5e-6, "mean_r": [0.02e-6], "gstdev": [1.4], "n_tot": [60.0e6]}}}}',
            outfreq = 100, 
            out_bin= '{"liq": {"rght": 1, "moms": [3], "drwt": "wet", "nbin": 1, "lnli": "lin", "left": 5e-20}}',
            outfile=outfile)
    parcel(**args)
    with netcdf.netcdf_file(outfile, 'r') as f:
        rv = np.array(f.variables['r_v'][:]).squeeze()
        z = np.array(f.variables['z'][:]).squeeze()
        T = np.array(f.variables['T'][:]).squeeze()
        if scheme.startswith("blk"):
            r_liq = np.array(f.variables['rc'][:]) + np.array(f.variables['rr'][:]).squeeze()
            r_ice = np.array(f.variables['ria'][:]) + np.array(f.variables['rib'][:]).squeeze()
        else: 
            r_liq = np.array(f.variables['liq_m3'][:]).squeeze() *4/3 * np.pi * common.rho_w
            r_ice = np.array(f.variables['ice_mix_ratio'][:]).squeeze()
        return z, rv, r_liq, r_ice, T

def test_plot_schemes():
    schemes = [("lgrngn", False), ("lgrngn", True), ("blk_1m", None)]
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1,4, figsize=(16, 6))
    for (scheme, time_dep) in schemes:
        z, rv, r_liq, r_ice, T = run_scheme(scheme, time_dep, f"test_{scheme}.nc")
        if scheme == "lgrngn" and time_dep == True:
            l = "lagranigan time-dep"
        elif time_dep == False:
            l = "lagranigan sigular"
        else:
            l = "1-moment bulk"
        ax[0].plot(rv*1e3, z, label=l, linestyle ='--' if scheme=='blk_1m' else '-')
        ax[1].plot(r_liq*1e3, z, label=l, linestyle ='--' if scheme=='blk_1m' else '-')
        ax[2].plot(r_ice*1e3, z, label=l, linestyle ='--' if scheme=='blk_1m' else '-')
        ax[3].plot(T, z, label=l, linestyle ='--' if scheme=='blk_1m' else '-')
        os.remove(f"test_{scheme}.nc")
        
    ax[0].set_ylabel("z [m]")
    ax[0].set_xlabel("$r_v$ [g/kg]")
    ax[1].set_xlabel("$r_{liq} [g/kg]$")
    ax[2].set_xlabel("$r_{ice} [g/kg]$")
    ax[3].set_xlabel("T [K]")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig("plots/outputs/plot_schemes_ice.svg")