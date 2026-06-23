#!/usr/bin/env python
import numpy as np
from libcloudphxx import lgrngn
from parcel_common import lognormal, sum_of_lognormals, _Chem_g_id, _Chem_a_id, _stats


def _micro_init(aerosol, opts, state):
  """Initialize the lagrangian microphysics scheme"""

  # lagrangian scheme options
  opts_init = lgrngn.opts_init_t()
  for opt in ["dt", "sd_conc", "chem_rho", "sstp_cond","ice_switch","time_dep_ice_nucl"]:
    setattr(opts_init, opt, opts[opt])
  opts_init.n_sd_max = opts_init.sd_conc
  if opts["rng_seed"] is not None:
      opts_init.rng_seed = int(opts["rng_seed"])

  opts_init.th_dry = True
  opts_init.const_p = False

  # read in the initial aerosol size distribution
  dry_distros = {}
  for name, dct in aerosol.items(): # loop over kappas
    lognormals = []
    for i in range(len(dct["mean_r"])):
      lognormals.append(lognormal(dct["mean_r"][i], dct["gstdev"][i], dct["n_tot"][i]))
    dry_distros[(float(dct["kappa"]), float(dct["sol_frac"]))] = sum_of_lognormals(lognormals)
  opts_init.dry_distros = dry_distros

  # better resolution for the SD tail
  if opts["large_tail"]:
      opts_init.sd_conc_large_tail = 1
      opts_init.n_sd_max = int(1e6)  # some more space for the tail SDs

  # switch off sedimentation and collisions
  opts_init.sedi_switch = False
  opts_init.coal_switch = False

  # switching on chemistry if either dissolving, dissociation or reactions are chosen
  opts_init.chem_switch = False
  if opts["chem_dsl"] or opts["chem_dsc"] or opts["chem_rct"]:
    opts_init.chem_switch = True
    opts_init.sstp_chem = opts["sstp_chem"]

  # initialisation
  micro = lgrngn.factory(lgrngn.backend_t.serial, opts_init)
  ambient_chem = {}
  if micro.opts_init.chem_switch:
    ambient_chem = dict((v, state[k]) for k,v in _Chem_g_id.items())
  micro.init(state["th_d"], state["r_v"], state["rhod"], ambient_chem=ambient_chem)

  return micro


def _micro_step(micro, state, info, opts):
  '''Microphysics step for lagrangian scheme'''
  libopts = lgrngn.opts_t()
  libopts.cond = True
  libopts.coal = False
  libopts.adve = False
  libopts.sedi = False
  libopts.ice_nucl = opts["ice_nucl"]
  libopts.depo = opts["depo"]

  # chemical options
  if micro.opts_init.chem_switch:
    # chem processes: dissolving, dissociation, reactions
    libopts.chem_dsl = opts["chem_dsl"]
    libopts.chem_dsc = opts["chem_dsc"]
    libopts.chem_rct = opts["chem_rct"]

  # get trace gases
  ambient_chem = {}
  if micro.opts_init.chem_switch:
    ambient_chem = dict((v, state[k]) for k,v in _Chem_g_id.items())

  # call libcloudphxx microphysics
  micro.step_sync(libopts, state["th_d"], state["r_v"], state["rhod"], ambient_chem=ambient_chem)
  micro.step_async(libopts)

  # update state after microphysics (needed for below update for chemistry)
  _stats(state, info)

  # update in state for aqueous chem (TODO do we still want to have aq chem in state?)
  if micro.opts_init.chem_switch:
    micro.diag_all() # selecting all particles
    for id_str, id_int in _Chem_g_id.items():
      # save changes due to chemistry
      micro.diag_chem(id_int)
      state[id_str.replace('_g', '_a')] = np.frombuffer(micro.outbuf())[0]
  if micro.opts_init.ice_switch:
    micro.diag_ice()
    micro.diag_ice_mix_ratio()
    state["ice_mix_ratio"] = np.frombuffer(micro.outbuf())[0]