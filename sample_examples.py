
import sys
import os
sys.path.append(os.path.abspath("/Users/chriszawora/Dropbox/WIEB/linprog/"))

from pathlib import Path
from itertools import product
from ghg_counterfactual2 import *
from tqdm import tqdm
from scipy.optimize import minimize


  
  
def run_all_optim(params):
  
  L1 = {
  'load': 100,
  'ghg_area': False,
  'name': "L1"
  }
  
  L2 = {
  'load': params[0],
  'ghg_area': False,
  'name': "L2"
  }
  
  L3 = {
  'load': 100,
  'ghg_area': True,
  'name': "L3"
  }
  
  loads = [L1, L2, L3]
  
  generators = make_generators(*params[1:len(params)])
  solution_set = run_all(generators, loads, "", write_tables = False)
  
  d1 = abs(solution_set['vistra_solution']['mc_energy'] - solution_set['caiso_solution']['mc_energy'])
  d2 = abs(solution_set['caiso_solution']['mc_energy'] - solution_set['noghg_solution']['mc_energy'])
  
  d3 = abs(solution_set['vistra_solution']['mc_ghg'] - solution_set['caiso_solution']['mc_ghg'])
  d4 = abs(solution_set['caiso_solution']['mc_ghg'] - solution_set['noghg_solution']['mc_ghg'])
  
  d5 = abs(solution_set['noghg_solution']['op_balance'])
  d6 = abs(solution_set['vistra_solution']['op_balance'])
  d7 = abs(solution_set['caiso_solution']['op_balance'])
  
  obj_value = -1* ((d1*d2 + d3*d4) * d5) + d6 * 1e10 + d7 * 1e10
  
  return obj_value


def make_generators(
  energy_cap1,
  energy_cap2,
  energy_cap3,
  energy_cap4,
  energy_bid1,
  energy_bid2,
  energy_bid3,
  energy_bid4,
  ghg_bid1,
  ghg_bid2,
  ghg_bid3,
  ghg_bid4
):
  
  gen1 = {
    "energy_bid": energy_bid1,
    "energy_cap": energy_cap1,
    "ghg_adder": ghg_bid1,
    "attrib_cap": energy_cap1,
    "ghg_area": False,
    "name": "gen1"
  }
  
  gen2 = {
    "energy_bid": energy_bid2,
    "energy_cap": energy_cap2,
    "ghg_adder": ghg_bid2,
    "attrib_cap": energy_cap2,
    "ghg_area": False,
    "name": "gen2"
  }
  
  gen3 = {
    "energy_bid": energy_bid3,
    "energy_cap": energy_cap3,
    "ghg_adder": ghg_bid3,
    "attrib_cap": energy_cap3,
    "ghg_area": False,
    "name": "gen3"
  }
  
  gen4 = {
    "energy_bid": energy_bid4,
    "energy_cap": energy_cap4,
    "ghg_adder": ghg_bid4,
    "attrib_cap": energy_cap4,
    "ghg_area": True,
    "name": "gen4"
  }

  generators = [gen1, gen2, gen3, gen4]

  return generators

#fixed load

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
  
  
loads = [L1, L2, L3]


x0 =  [
  30,
  200,
  95,
  140,
  100,
  8,
  10,
  2,
  100,
  5,
  1,
  0,
  100]
  
res = minimize(run_all_optim, x0)

L2 = {
  'load': res.x[0],
  'ghg_area': False,
  'name': "L2"
  }

generators = make_generators(*[float(i) for i in res.x[1:len(res.x)]])
loads = [L1, L2, L3]

outdir = "/Users/chriszawora/Dropbox/WIEB/linprog/example01"
Path(outdir).mkdir(exist_ok=True)

x= run_all(generators, loads, outdir)





energy_prices = [i for i in range(0, 30, 5)]
adder_prices = [i for i in range(0, 15, 3)]


param_iter = product(energy_prices, energy_prices, energy_prices, energy_prices,
            adder_prices, adder_prices, adder_prices, adder_prices)


cols = [
  "energy_bid1",
  "energy_bid2",
  "energy_bid3",
  "energy_bid4",
  "ghg_bid1",
  "ghg_bid2",
  "ghg_bid3",
  "ghg_bid4",
  "market_operator_position_vistra",
  "market_operator_position_caiso",
  "market_operator_position_noghg",
  "mc_energy_vistra",
  "mc_energy_caiso",
  "mc_energy_noghg",
  "mc_ghg_vistra",
  "mc_ghg_caiso",
  "mc_ghg_noghg",
  "gen1_profit_vistra",
  "gen1_profit_caiso",
  "gen1_profit_noghg",
  "gen2_profit_vistra",
  "gen2_profit_caiso",
  "gen2_profit_noghg",
  "gen3_profit_vistra",
  "gen3_profit_caiso",
  "gen3_profit_noghg",
  "gen4_profit_vistra",
  "gen4_profit_caiso",
  "gen4_profit_noghg"
]

outfile = open("/Users/chriszawora/Dropbox/WIEB/linprog/example_sample/sample_results.csv", "a")
outfile.write(",".join(cols) + "\n")

for param_set in tqdm(param_iter):
  
  if len(set(param_set[0:4])) != 4:
    continue
  
  if len(set(param_set[4:8])) != 4:
    continue
                
  generators = make_generators(*param_set)
  solutions = run_all(generators, loads, "", write_tables = False)
  
  log_str = []
            
  log_str.append(param_set[0])    
  log_str.append(param_set[1])     
  log_str.append(param_set[2])      
  log_str.append(param_set[3])      

  log_str.append(param_set[4])      
  log_str.append(param_set[5])      
  log_str.append(param_set[6])      
  log_str.append(param_set[7])      

  log_str.append(solutions['vistra_solution']['op_balance'])
  log_str.append(solutions['caiso_solution']['op_balance'])
  log_str.append(solutions['noghg_solution']['op_balance'])

  log_str.append(solutions['vistra_solution']['mc_energy'])
  log_str.append(solutions['caiso_solution']['mc_energy'])
  log_str.append(solutions['noghg_solution']['mc_energy'])
  
  log_str.append(solutions['vistra_solution']['mc_ghg'])
  log_str.append(solutions['caiso_solution']['mc_ghg'])
  log_str.append(solutions['noghg_solution']['mc_ghg'])
  
  for g in [1, 2, 3, 4]:
    
    log_str.append(
      solutions['vistra_solution'][f'gen{g}_revenue_res'] - solutions['vistra_solution'][f'gen{g}_prod_res']
      )
      
    log_str.append(
      solutions['caiso_solution'][f'gen{g}_revenue_res'] - solutions['caiso_solution'][f'gen{g}_prod_res']
      )

    log_str.append(
      solutions['noghg_solution'][f'gen{g}_revenue_res'] - solutions['noghg_solution'][f'gen{g}_prod_res']
      )
  
  log_str = [str(i) for i in log_str]
  outfile.write(",".join(log_str) + "\n")
  
outfile.close()
