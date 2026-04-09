"""
This test runs the parcel model using blk_1m microphysics scheme with ice.
It checks that the final values of ria and rc match their reference values (are consistent in each run).
"""

import sys, subprocess
sys.path.insert(0, "../")
sys.path.insert(0, "./")
from parcel import parcel
from scipy.io import netcdf
import numpy as np

def test_bulk_ice():
    outfile = "test_blk_ice.nc"
    args = dict(
        dt=0.1,
        z_max=500,
        w=1.0,
        T_0=273,
        RH_0 = 1,
        outfile=outfile,
        outfreq=100,
        scheme="blk_1m",
        ice_switch=True
    )
    parcel(**args)
    with netcdf.netcdf_file(outfile, 'r') as f:
        ria = np.array(f.variables['ria'][:])
        rc = np.array(f.variables['rc'][:])
        f.close()
    assert np.isclose(ria[-1], 8.9947e-11, rtol=1e-3)
    assert np.isclose(rc[-1], 6.0941e-4, rtol=1e-3)
    subprocess.call(["rm", outfile])