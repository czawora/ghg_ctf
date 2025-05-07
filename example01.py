
import sys
import os
sys.path.append(os.path.abspath("/Users/chriszawora/Dropbox/WIEB/linprog/"))

from pathlib import Path
from ghg_counterfactual2 import *


L1 = make_load(load = 100, ghg_area = False, BAA = "A", name = "L1")
L2 = make_load(load = 40, ghg_area = False, BAA = "B", name = "L2")
L3 = make_load(load = 90, ghg_area = True, BAA = "C", name = "L3")
L3_mg = L3 #make_load(load = 90, ghg_area = True, BAA = "C", name = "L3")
  
gen1 = make_generator(
  energy_bid = 17, energy_cap = 50, ghg_adder = 4,
  ghg_area = False, name = "G1", BAA = "A"
  )
gen2 = make_generator(
  energy_bid = 10, energy_cap = 150, ghg_adder = 0,
  ghg_area = False, name = "G2", BAA = "A"
  )
gen3 = make_generator(
  energy_bid = 12, energy_cap = 50, ghg_adder = 13,
  ghg_area = False, name = "G3", BAA = "B"
  )
gen4 = make_generator(
  energy_bid = 13, energy_cap = 100, ghg_adder = 20,
  ghg_area = True, name = "G4", BAA = "C"
  )

generators = [gen1, gen2, gen3, gen4]
ctf_loads = [L1, L2, L3]
market_loads = [L1, L2, L3]

outdir = "/Users/chriszawora/Dropbox/WIEB/linprog/example01"
Path(outdir).mkdir(exist_ok=True)

x = run_all(generators, ctf_loads, market_loads, outdir)


