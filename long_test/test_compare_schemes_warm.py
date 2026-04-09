"""
This test runs the parcel model (with T > 0 C) using three different microphysics schemes:
- lgrngn (Lagrangian particle-based)
- blk_1m (bulk warm)
- blk_1m_ice (bulk ice)

It compares the final values of rv, th_d, and total condensed water in different schemes. 
"""

import sys, os
sys.path.insert(0, "../")
sys.path.insert(0, "./")

import numpy as np
from parcel import parcel
from scipy.io import netcdf
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
            r_tot = np.array(f.variables['radius_m3'][:]) *4/3 * np.pi * common.rho_w
    return rv, th_d, r_tot, z

def test_compare_schemes():
    schemes = ["lgrngn", "blk_1m", "blk_1m_ice"]
    results = {}
    for scheme in schemes:
        if scheme == "blk_1m_ice":
            rv, th_d, r_tot, z = run_scheme("blk_1m", True, f"test_{scheme}.nc")
        else:
            rv, th_d, r_tot, z = run_scheme(scheme, False, f"test_{scheme}.nc")
        results[scheme] = (rv, th_d, r_tot, z)
        os.remove(f"test_{scheme}.nc")

    # Compare final values
    rv_vals = [results[s][0][-1] for s in schemes]
    th_d_vals = [results[s][1][-1] for s in schemes]
    r_tot_vals = [results[s][2][-1] for s in schemes]
    
    # Check closeness
    for i in range(1, len(schemes)):
      assert np.isclose(rv_vals[0], rv_vals[i], rtol=5e-4), f"r_v differs: {rv_vals[0]} vs {rv_vals[i]}"
      assert np.isclose(th_d_vals[0], th_d_vals[i], rtol=5e-4), f"th_d differs: {th_d_vals[0]} vs {th_d_vals[i]}"
      assert np.isclose(r_tot_vals[0], r_tot_vals[i], rtol=1e-2), f"r_tot differs: {r_tot_vals[0]} vs {r_tot_vals[i]}"