#!/usr/bin/env python
from scipy.io import netcdf
import numpy as np
import json
from parcel_common import _Chem_g_id, _Chem_a_id


def _output_bins(fout, t, micro, opts, spectra):
  for dim, dct in spectra.items():
    for bin in range(dct["nbin"]):
      if dct["drwt"] == 'wet':
        micro.diag_wet_rng(
          fout.variables[dim+"_r_wet"][bin],
          fout.variables[dim+"_r_wet"][bin] + fout.variables[dim+"_dr_wet"][bin]
        )
      elif dct["drwt"] == 'dry':
        micro.diag_dry_rng(
          fout.variables[dim+"_r_dry"][bin],
          fout.variables[dim+"_r_dry"][bin] + fout.variables[dim+"_dr_dry"][bin]
        )
      elif dct["drwt"] == 'ice_a':
        micro.diag_ice_a_rng(
          fout.variables[dim+"_r_ice_a"][bin],
          fout.variables[dim+"_r_ice_a"][bin] + fout.variables[dim+"_dr_ice_a"][bin]
        )
      elif dct["drwt"] == 'ice_c':
        micro.diag_ice_c_rng(
          fout.variables[dim+"_r_ice_c"][bin],
          fout.variables[dim+"_r_ice_c"][bin] + fout.variables[dim+"_dr_ice_c"][bin]
        )
      else: raise Exception("drwt should be wet or dry or ice_a or ice_c")

      for vm in dct["moms"]:
        if type(vm) == int:
          # calculating moments
          if dct["drwt"] == 'wet':
            micro.diag_wet_mom(vm)
          elif dct["drwt"] == 'dry':
            micro.diag_dry_mom(vm)
          elif dct["drwt"] == 'ice_a':
            micro.diag_ice_a_mom(vm)
          elif dct["drwt"] == 'ice_c':
            micro.diag_ice_c_mom(vm)
          else: raise Exception("drwt should be wet or dry or ice_a or ice_c")
          fout.variables[dim+'_m'+str(vm)][int(t), int(bin)] = np.frombuffer(micro.outbuf())
        else:
          # calculate chemistry
          micro.diag_chem(_Chem_a_id[vm])
          fout.variables[dim+'_'+vm][int(t), int(bin)] = np.frombuffer(micro.outbuf())


def _output_init(micro, opts, spectra):
  """Initialize output file for lagrangian scheme"""
  # file & dimensions
  fout = netcdf.netcdf_file(opts["outfile"], 'w')
  fout.createDimension('t', None)
  for name, dct in spectra.items():
    fout.createDimension(name, dct["nbin"])

    tmp = name + '_r_' + dct["drwt"]
    fout.createVariable(tmp, 'd', (name,))
    fout.variables[tmp].unit = "m"
    fout.variables[tmp].description = "left bin edge"

    tmp = name + '_dr_' + dct["drwt"]
    fout.createVariable(tmp, 'd', (name,))
    fout.variables[tmp].unit = "m"
    fout.variables[tmp].description = "bin width"

    if dct["lnli"] == 'log':
      from math import exp, log
      dlnr = (log(dct["rght"]) - log(dct["left"])) / dct["nbin"]
      allbins = np.exp(log(dct["left"]) + np.arange(dct["nbin"]+1) * dlnr)
      fout.variables[name+'_r_'+dct["drwt"]][:] = allbins[0:-1]
      fout.variables[name+'_dr_'+dct["drwt"]][:] = allbins[1:] - allbins[0:-1]
    elif dct["lnli"] == 'lin':
      dr = (dct["rght"] - dct["left"]) / dct["nbin"]
      fout.variables[name+'_r_'+dct["drwt"]][:] = dct["left"] + np.arange(dct["nbin"]) * dr
      fout.variables[name+'_dr_'+dct["drwt"]][:] = dr
    else: raise Exception("lnli should be log or lin")

    for vm in dct["moms"]:
      if (vm in _Chem_a_id):
        fout.createVariable(name+'_'+vm, 'd', ('t',name))
        fout.variables[name+'_'+vm].unit = 'kg of chem species dissolved in cloud droplets (kg of dry air)^-1'
      else:
        assert(type(vm)==int)
        fout.createVariable(name+'_m'+str(vm), 'd', ('t',name))
        fout.variables[name+'_m'+str(vm)].unit = 'm^'+str(vm)+' (kg of dry air)^-1'

  units = {"z"  : "m",     "t"   : "s",     "r_v"  : "kg/kg", "th_d" : "K", "rhod" : "kg/m3",
           "p"  : "Pa",    "T"   : "K",     "RH"   : "1",    "T_blk"   : "K",     "RH_blk"   : "1"
  }

  if micro.opts_init.chem_switch:
    for id_str in _Chem_g_id.keys():
      units[id_str] = "gas mixing ratio [kg / kg dry air]"
      units[id_str.replace('_g', '_a')] = "kg of chem species (both undissociated and ions) dissolved in cloud droplets (kg of dry air)^-1"

  for var_name, unit in units.items():
    fout.createVariable(var_name, 'd', ('t',))
    fout.variables[var_name].unit = unit

  if micro.opts_init.ice_switch:
    fout.createVariable("ice_mix_ratio", 'd', ('t',))
    fout.variables["ice_mix_ratio"].unit = "kg/kg"

  return fout


def _output_init_blk_1m(opts):
    """Initialize output file for bulk microphysics scheme without ice processes"""
    fout = netcdf.netcdf_file(opts["outfile"], 'w')
    fout.createDimension('t', None)
    
    # Basic variables with their units
    vars_units = {
        "z": ("m",), 
        "t": ("s",),
        "r_v": ("kg/kg",),
        "rc": ("kg/kg",),
        "rr": ("kg/kg",),
        "th_d": ("K",),
        "rhod": ("kg/m3",),
        "p": ("Pa",),
        "T": ("K",),
        "RH": ("1",),
        "T_blk": ("K",),
        "RH_blk": ("1",)
    }
    
    for var, (unit,) in vars_units.items():
        fout.createVariable(var, 'd', ('t',))
        fout.variables[var].unit = unit
        
    return fout


def _output_init_blk_1m_ice(opts):
    """Initialize output file for bulk ice microphysics scheme"""
    fout = netcdf.netcdf_file(opts["outfile"], 'w')
    fout.createDimension('t', None)
    
    # Basic variables with their units
    vars_units = {
        "z": ("m",), 
        "t": ("s",),
        "r_v": ("kg/kg",),
        "rc": ("kg/kg",),
        "rr": ("kg/kg",),
        "ria": ("kg/kg",),
        "rib": ("kg/kg",),
        "th_d": ("K",),
        "rhod": ("kg/m3",),
        "p": ("Pa",),
        "T": ("K",),
        "RH": ("1",),
        "T_blk": ("K",),
        "RH_blk": ("1",)
    }
    
    for var, (unit,) in vars_units.items():
        fout.createVariable(var, 'd', ('t',))
        fout.variables[var].unit = unit
        
    return fout


def _output_save(fout, state, rec):
  for var, val in state.items():
    fout.variables[var][int(rec)] = val


def _save_attrs(fout, dictnr):
  for var, val in dictnr.items():
    setattr(fout, var, val)


def _output(fout, opts, micro, state, rec, spectra):
  _output_bins(fout, rec, micro, opts, spectra)
  _output_save(fout, state, rec)
