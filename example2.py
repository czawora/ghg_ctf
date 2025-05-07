
import sys
import os
sys.path.append(os.path.abspath("/Users/chriszawora/Dropbox/WIEB/linprog/"))

from pathlib import Path
from ghg_counterfactual2 import *


L1 = {
  'load': 100,
  'ghg_area': False,
  'name': "L1"
  }
  
L2 = {
  'load': 30,
  'ghg_area': False,
  'name': "L2"
  }
  
L3 = {
  'load': 100,
  'ghg_area': True,
  'name': "L3"
  }

gen1 = {
  "energy_bid": 8,
  "energy_cap": 200,
  "ghg_adder": 5,
  "attrib_cap": 200,
  "ghg_area": False,
  "name": "gen1"
}

gen2 = {
  "energy_bid": 10,
  "energy_cap": 95,
  "ghg_adder": 1,
  "attrib_cap": 95,
  "ghg_area": False,
  "name": "gen2"
}

gen3 = {
  "energy_bid": 2,
  "energy_cap": 140,
  "ghg_adder": 0,
  "attrib_cap": 140,
  "ghg_area": False,
  "name": "gen3"
}

gen4 = {
  "energy_bid": 100,
  "energy_cap": 100,
  "ghg_adder": 10,
  "attrib_cap": 100,
  "ghg_area": True,
  "name": "gen4"
}

generators = [gen1, gen2, gen3, gen4]
loads = [L1, L2, L3]

outdir = "/Users/chriszawora/Dropbox/WIEB/linprog/example_ppt"
Path(outdir).mkdir(exist_ok=True)

run_all(generators, loads, outdir)
