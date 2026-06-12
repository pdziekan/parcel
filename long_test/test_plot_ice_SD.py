"""
This test runs the parcel model with lagrangian ice microphysics, 
with homogeneous and heterogeneous time-dependent freezing.
Mixing ratios of ice, liquid, and water vapor are plotted.
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

def plot_profiles(fnc, output_name):
    plt.clf()
    plt.rcParams.update({'font.size': 14})
    fig, plots = plt.subplots(1, 2, figsize=(15, 5))
    plots[0].set_xlabel('mixing ratio [g/kg]')
    plots[1].set_xlabel('T [C]')
    for ax in plots:
        ax.set_ylabel('z [km]')
        ax.grid()

    z = fnc.variables["z"][:] / 1000 #km
    r_v = fnc.variables["r_v"][:] * 1000  #g/kg
    r_liq = np.array([i[0] for i in fnc.variables['liq_m3'][:]]) *4/3 * np.pi * common.rho_w * 1000 #g/kg
    r_ice = fnc.variables["ice_mix_ratio"][:] * 1000 #g/kg

    plots[0].plot(r_v + r_liq + r_ice, z)
    plots[0].plot(r_v, z)
    plots[0].plot(r_liq, z)
    plots[0].plot(r_ice, z)
    plots[0].legend(['$r_{tot}$', '$r_v$', '$r_{liq}$', '$r_{ice}$'], loc='best')
    plots[1].plot(fnc.variables["T"][:] - 273.15, z)
    plt.suptitle("Homogeneous ice nucleation" if output_name == "ice_SD_plot_hom.svg" else "Heterogeneous ice nucleation")

    if not os.path.exists("plots/outputs/"):
        subprocess.call(["mkdir", "plots/outputs/"])
    plt.savefig(os.path.join("plots/outputs/", output_name))

def test_plot_ice_SD():   
    for (rd_insol, output_name) in [("0.5e-6", "ice_SD_plot_het.svg"), ("0", "ice_SD_plot_hom.svg")]:
        outfile = "onesim_plot.nc"
        parcel(dt=1.,w=1.,sd_conc=100,
            z_max = 5000.0,
            T_0 = 263.0,
            RH_0 = 1.,
            scheme = "lgrngn",
            ice_switch=True,
            depo = True,
            ice_nucl=True,
            time_dep_ice_nucl=True,
            aerosol = f'{{"ammonium_sulfate": {{"kappa": 0.61, "rd_insol": {rd_insol}, "mean_r": [0.02e-6], "gstdev": [1.4], "n_tot": [60.0e6]}}}}',
            outfreq = 100, 
            out_bin= '{"liq": {"rght": 1, "moms": [0,3], "drwt": "wet", "nbin": 1, "lnli": "lin", "left": 5e-20}}',
            outfile=outfile)
        fnc = netcdf.netcdf_file(outfile)
        plot_profiles(fnc, output_name)
        fnc.close()
        subprocess.call(["rm", outfile])