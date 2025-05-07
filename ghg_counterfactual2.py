
from great_tables import GT, md, html, google_font, style, loc
import pandas as pd
from copy import deepcopy
from ortools.linear_solver import pywraplp


def add_settlements(generators, loads, solution):
  
  mc_energy = solution["power_balance_constraint"] + solution["export_constraint"]
  mc_ghg = -1*solution["export_constraint"]
  mc_both = mc_energy + mc_ghg

  solution['mc_energy'] = mc_energy
  solution['mc_ghg'] = mc_ghg
  
  op_balance = 0

  for gen in generators:
    
    energy_name = f'{gen["name"]}_energy'
    attrib_name = f'{gen["name"]}_attrib'
  
    solution[f'{gen["name"]}_revenue_form'] = f'({solution[energy_name]} * ${mc_energy}) + ({solution[attrib_name]} * ${mc_ghg})'
    solution[f'{gen["name"]}_revenue_res'] = solution[energy_name] * mc_energy + solution[attrib_name] * mc_ghg
    
    op_balance += (solution[energy_name] * mc_energy + solution[attrib_name] * mc_ghg)

    solution[f'{gen["name"]}_prod_form'] = f'({solution[energy_name]} * ${gen["energy_bid"]}) + ({solution[attrib_name]} * ${gen["ghg_adder"]})'
    solution[f'{gen["name"]}_prod_res'] = solution[energy_name] * gen["energy_bid"] + solution[attrib_name] * gen["ghg_adder"]
  
  for load in loads:
    
    if load['ghg_area']:
      
      solution[f'{load["name"]}_cost_form'] = f'{load["load"]} * ${mc_both}'
      solution[f'{load["name"]}_cost_res'] = load['load'] * mc_both
      
      op_balance -= (load['load'] * mc_both)

    else:
      
      solution[f'{load["name"]}_cost_form'] = f'{load["load"]} * ${mc_energy}'
      solution[f'{load["name"]}_cost_res'] = load['load'] * mc_energy
  
      op_balance -= (load['load'] * mc_energy)

  solution['op_balance'] = op_balance
  
  return solution



def run_solver(
  generators, 
  loads, 
  include_ghg = True, 
  net_import_reference_mw = 0
  ):
  
  non_ghg_area_load = sum([l['load'] for l in loads if not l['ghg_area']]) 
  ghg_area_load = sum([l['load'] for l in loads if l['ghg_area']]) 
  total_load = ghg_area_load + non_ghg_area_load

  gen_energy_costs = [g['energy_bid'] for g in generators]
  gen_ghg_adders = [g['ghg_adder'] for g in generators]
  
  #get solver 
  solver = pywraplp.Solver.CreateSolver('GLOP')

  # declare decision variables
  gen_energy_vars = []
  gen_attrib_vars = []
  obj_func_parts = []

  for idx, gen in enumerate(generators):
    
    gen_energy_vars.append(solver.NumVar(0.0, gen['energy_cap'], f'{gen["name"]}_energy'))

    if include_ghg:
      gen_attrib_vars.append(solver.NumVar(0.0, gen['attrib_cap'], f'{gen["name"]}_attrib'))
    
      if gen['ghg_area']:
        obj_func_parts.append(gen_energy_vars[idx] * (gen['energy_bid'] + gen['ghg_adder']))
      
      else:
        obj_func_parts.append(gen_energy_vars[idx] * gen['energy_bid'])
        obj_func_parts.append(gen_attrib_vars[idx] * gen['ghg_adder'])
        
    else:
      obj_func_parts.append(gen_energy_vars[idx] * gen['energy_bid'])

  obj_func = sum(obj_func_parts)
  
  solver.Minimize(obj_func)

  # declare constraints
  constraints = {}
  
  constraints.update({
    solver.Add(sum(gen_energy_vars) == total_load).name(): "power_balance_constraint"
  })
  
  if include_ghg:
    for idx, vars in enumerate(zip(gen_energy_vars, gen_attrib_vars)):
      
      if generators[idx]['ghg_area']:
        constraints.update({solver.Add(vars[1] == 0).name(): f"alloc_zero{idx + 1}"})
      
      if not generators[idx]['ghg_area']:
        constraints.update({solver.Add(vars[1] - vars[0] <= 0).name(): f"alloc_limit{idx + 1}"})
    
    constraints.update({
      solver.Add(
        sum(gen_energy_vars[0:3]) + (-1) * sum(gen_attrib_vars[0:3]) <= non_ghg_area_load + net_import_reference_mw).name():
        "export_constraint"
      })
      
  results = solver.Solve()
  
  if results != pywraplp.Solver.OPTIMAL:
    raise("Solution Not Optimal")
  
  solution = {}
  
  for var in solver.variables():
    solution.update({var.name(): round(var.solution_value())})
    
  for ct in solver.constraints():
    solution.update({constraints[ct.name()]: round(ct.dual_value())})
    
  return solver, solution


def make_ctf_table(
  generators,
  gen_idx_baa_dict, 
  vistra_ctf_dispatch, 
  caiso_ctf_dispatch, 
  noghg_ctf_dispatch, 
  outfile
  ):
    
  ctf_list = [vistra_ctf_dispatch, caiso_ctf_dispatch, noghg_ctf_dispatch]
  
  ctf_table = {
    "Counterfactual": ["Vistra", "CAISO", "No GHG Cost"]
  }
  
  for baa in sorted(gen_idx_baa_dict):
    
    baa_ctf_list = []
    
    for ctf in ctf_list:
      baa_ctf_list.append("\n".join([f'{generators[idx]["name"]}: {ctf[idx]} MW' for idx in gen_idx_baa_dict[baa]]))
    
    ctf_table.update({
      f'BAA {baa}': baa_ctf_list
    })
  
  GT(pd.DataFrame(ctf_table))\
  .tab_header(
        title=md("**Counterfactual Dispatch**")
    )\
  .tab_style(
      style=style.text(whitespace = "pre-wrap"),
      locations=loc.body()
      ) \
  .tab_style(
      style=style.fill(color="#357da8"),
      locations=loc.column_labels()
      ) \
  .tab_style(
      style=style.fill(color="#CBD1D6"), 
      locations=loc.body(rows = [0, 2])
      ) \
  .tab_style(
      style=style.fill(color="#E3E6E9"),
      locations=loc.body(rows = [1])
      ) \
  .tab_style(
      style=style.text(weight = 1000, color = "white"),
      locations=loc.column_labels()
      ) \
  .tab_style(
      style=style.text(weight = "bolder"),
      locations=loc.body(columns = 0)
      ) \
  .tab_style(
    style=style.borders(color="#FFFFFF", weight = 2),
    locations=loc.body()
  ) \
  .tab_style(
    style=style.borders(color="#FFFFFF", weight = 2),
    locations=loc.column_labels()
  ) \
  .cols_align(    
      align='center'
      ) \
  .opt_table_outline( style='solid', width='3px', color='#FFFFFF') \
  .opt_table_font(font=google_font("Roboto Slab SemiBold 600")) \
  .save(outfile)
  
  
def make_market_table(
  generators,
  gen_idx_baa_dict, 
  non_ghg_baas,
  ctf_dispatch, 
  solution, 
  outfile):
  
  mc_energy = solution["mc_energy"] 
  mc_ghg = solution["mc_ghg"]
  
  table = {
  "": ["Counterfactual Dispatch", "Market Dispatch", "GHG Attribution", "Marginal Cost, Energy\n", "Marginal Cost, GHG\n", "LMP\n"],
  }
  
  for baa in sorted(gen_idx_baa_dict):
    
    baa_gen_name_idx_pair = [(generators[idx]["name"], idx) for idx in gen_idx_baa_dict[baa]]
    
    col_list = []
    col_list.append("\n".join([f'{name}: {ctf_dispatch[idx]} MW' for name, idx in baa_gen_name_idx_pair]))
    col_list.append("\n".join([f'{name}: {solution[f"{name}_energy"]} MW' for name, idx in baa_gen_name_idx_pair]))
    col_list.append("\n".join([f'{name}: {solution[f"{name}_attrib"]} MW' for name, idx in baa_gen_name_idx_pair]))
    col_list.append(f'${mc_energy}')
    
    if baa in non_ghg_baas:
      col_list.append(f'$0')
      col_list.append(f'${mc_energy}')
    else:
      col_list.append(f'${mc_ghg}')
      col_list.append(f'${mc_energy + mc_ghg}')

    table.update({
      f'BAA {baa}': col_list
    })
  
  
  GT(pd.DataFrame(table))\
  .tab_style(
      style=style.text(whitespace = "pre-wrap"),
      locations=loc.body()
      ) \
  .tab_style(
      style=style.fill(color="#357da8"),
      locations=loc.column_labels()
      ) \
  .tab_style(
      style=style.fill(color="#CBD1D6"), 
      locations=loc.body(rows = [0, 2, 4])
      ) \
  .tab_style(
      style=style.fill(color="#E3E6E9"),
      locations=loc.body(rows = [1, 3, 5])
      ) \
  .tab_style(
      style=style.text(weight = 1000, color = "white"),
      locations=loc.column_labels()
      ) \
  .tab_style(
      style=style.text(weight = "bolder"),
      locations=loc.body(columns = 0)
      ) \
  .tab_style(
    style=style.borders(color="#FFFFFF", weight = 2),
    locations=loc.body()
  ) \
  .tab_style(
    style=style.borders(color="#FFFFFF", weight = 2),
    locations=loc.column_labels()
  ) \
  .cols_width(
    cases={
        "": "37%",
        "BAA A": "21%",
        "BAA B": "21%",
        "BAA C": "21%"
    }
  )\
  .cols_align(    
      align='center'
      ) \
  .opt_table_outline( style='solid', width='3px', color='#FFFFFF') \
  .opt_table_font(font=google_font("Roboto Slab SemiBold 600")) \
  .tab_options(table_width = "500px") \
  .save(outfile)



def make_gen_settlement_table(gen, vistra_solution, caiso_solution, noghg_solution):
  
  energy_name = f'{gen["name"]}_energy'
  attrib_name = f'{gen["name"]}_attrib'
  
  vistra_revenue_form = vistra_solution[f'{gen["name"]}_revenue_form']
  vistra_revenue_res = vistra_solution[f'{gen["name"]}_revenue_res']
  vistra_prod_form = vistra_solution[f'{gen["name"]}_prod_form']
  vistra_prod_res = vistra_solution[f'{gen["name"]}_prod_res']
  
  caiso_revenue_form = caiso_solution[f'{gen["name"]}_revenue_form']
  caiso_revenue_res = caiso_solution[f'{gen["name"]}_revenue_res']
  caiso_prod_form = caiso_solution[f'{gen["name"]}_prod_form']
  caiso_prod_res = caiso_solution[f'{gen["name"]}_prod_res']
  
  noghg_revenue_form = noghg_solution[f'{gen["name"]}_revenue_form']
  noghg_revenue_res = noghg_solution[f'{gen["name"]}_revenue_res']
  noghg_prod_form = noghg_solution[f'{gen["name"]}_prod_form']
  noghg_prod_res = noghg_solution[f'{gen["name"]}_prod_res']

  table = {
    "Counterfactual": ["Vistra", "CAISO", "No GHG Cost"],
    "Dispatch": [
      f'{vistra_solution[energy_name]} MW',
      f'{caiso_solution[energy_name]} MW',
      f'{noghg_solution[energy_name]} MW'
    ],
    "Attribution": [
      f'{vistra_solution[attrib_name]} MW',
      f'{caiso_solution[attrib_name]} MW',
      f'{noghg_solution[attrib_name]} MW'
    ],
    "Revenue": [
      f'{vistra_revenue_form} = ${vistra_revenue_res}',
      f'{caiso_revenue_form} = ${caiso_revenue_res}',
      f'{noghg_revenue_form} = ${noghg_revenue_res}'
    ],
    "Production Cost": [
      f'{vistra_prod_form} = ${vistra_prod_res}',
      f'{caiso_prod_form} = ${caiso_prod_res}',
      f'{noghg_prod_form} = ${noghg_prod_res}'
    ],
    "Net": [
      f'${vistra_revenue_res - vistra_prod_res}',
      f'${caiso_revenue_res - caiso_prod_res}',
      f'${noghg_revenue_res - noghg_prod_res}'
    ]
  }
    
  gt_table = GT(pd.DataFrame(table))
    
  return gt_table
  
  
def make_load_settlement_table(loads, vistra_solution, caiso_solution, noghg_solution):
  
  sols = [vistra_solution, caiso_solution, noghg_solution]
  
  table = {"Counterfactual": ["Vistra", "CAISO", "No GHG Cost"]}
  
  for l in loads:
    load_name = l["name"]
    form = f"{load_name}_cost_form"
    res = f"{load_name}_cost_res"
    table.update({load_name: [f'{s[form]} = ${s[res]}' for s in sols]})
  
  gt_table = GT(pd.DataFrame(table))
    
  return gt_table


def make_op_settlement_table(generators, loads, vistra_solution, caiso_solution, noghg_solution):

  row_labels = [g["name"] for g in generators] + [l["name"] for l in loads] + ["ISO Balance"]

  vistra_gen_payments = [-1*vistra_solution[f'{g["name"]}_revenue_res'] for g in generators] 
  vistra_load_payments = [vistra_solution[f'{l["name"]}_cost_res'] for l in loads]
  vistra_op_balance = sum(vistra_gen_payments) + sum(vistra_load_payments)
    
  caiso_gen_payments = [-1*caiso_solution[f'{g["name"]}_revenue_res'] for g in generators] 
  caiso_load_payments = [caiso_solution[f'{l["name"]}_cost_res'] for l in loads]
  caiso_op_balance = sum(caiso_gen_payments) + sum(caiso_load_payments)

  noghg_gen_payments = [-1*noghg_solution[f'{g["name"]}_revenue_res'] for g in generators] 
  noghg_load_payments = [noghg_solution[f'{l["name"]}_cost_res'] for l in loads]
  noghg_op_balance = sum(noghg_gen_payments) + sum(noghg_load_payments)
    

  table = {
    "": row_labels,
    "Vistra": vistra_gen_payments + vistra_load_payments + [vistra_op_balance],
    "CAISO": caiso_gen_payments + caiso_load_payments + [caiso_op_balance],
    "No GHG Cost": noghg_gen_payments + noghg_load_payments + [noghg_op_balance]
  }
  
  gt_table = GT(pd.DataFrame(table))
    
  return gt_table


def make_lmp_table(vistra_solution, caiso_solution, noghg_solution):
  
  table = {
    "Counterfactual": ["Vistra", "CAISO", "No GHG Cost"],
    "BAA A": [
      f'${vistra_solution["mc_energy"]}',
      f'${caiso_solution["mc_energy"]}',
      f'${noghg_solution["mc_energy"]}'
    ],
    "BAA B": [
      f'${vistra_solution["mc_energy"]}',
      f'${caiso_solution["mc_energy"]}',
      f'${noghg_solution["mc_energy"]}'
    ],
    "BAA C":  [
      f'${vistra_solution["mc_energy"] + vistra_solution["mc_ghg"]}',
      f'${caiso_solution["mc_energy"] + caiso_solution["mc_ghg"]}',
      f'${noghg_solution["mc_energy"] + noghg_solution["mc_ghg"]}'
    ] 
  }
  
  gt_table = GT(pd.DataFrame(table))
    
  return gt_table


def make_rev_dist_table(
  generators, 
  vistra_solution, 
  caiso_solution, 
  noghg_solution, 
  outfile):
    
  vistra_gen_payments_total = sum([vistra_solution[f'{g["name"]}_revenue_res'] for g in generators])
  vistra_gen_payments = [vistra_solution[f'{g["name"]}_revenue_res'] for g in generators] 
  vistra_gen_payments_pct = [round(vistra_solution[f'{g["name"]}_revenue_res']/vistra_gen_payments_total, 2) for g in generators] 

  caiso_gen_payments_total = sum([caiso_solution[f'{g["name"]}_revenue_res'] for g in generators] )
  caiso_gen_payments = [caiso_solution[f'{g["name"]}_revenue_res'] for g in generators] 
  caiso_gen_payments_pct = [round(caiso_solution[f'{g["name"]}_revenue_res']/caiso_gen_payments_total, 2) for g in generators] 

  noghg_gen_payments_total = sum([noghg_solution[f'{g["name"]}_revenue_res'] for g in generators] )
  noghg_gen_payments = [noghg_solution[f'{g["name"]}_revenue_res'] for g in generators] 
  noghg_gen_payments_pct = [round(noghg_solution[f'{g["name"]}_revenue_res']/noghg_gen_payments_total, 2) for g in generators] 

  table = {
    "Counterfactual": ["Vistra", "CAISO", "No GHG Cost"],
    "Total": [
      f'${vistra_gen_payments_total}',
      f'${caiso_gen_payments_total}',
      f'${noghg_gen_payments_total}'
    ]}
    
  for idx, gen in enumerate(generators):
    
    table.update({
      gen["name"]: [
        f'{vistra_gen_payments_pct[idx]} (${vistra_gen_payments[idx]})',
        f'{caiso_gen_payments_pct[idx]} (${caiso_gen_payments[idx]})',
        f'{noghg_gen_payments_pct[idx]} (${noghg_gen_payments[idx]})'
        ]
    })
    
  gt_table = GT(pd.DataFrame(table))
  gt_table.save(outfile)
  

def run_all(generators, ctf_loads, market_loads, outdir, write_tables = True):

  # note assuming gen and load is in baa order with ghg baa last

  baa_set = sorted(list(set([i['BAA'] for i in generators])))
  # assuming generator baas cover all baas

  # group gen by baa
  gen_baa_dict = {baa: [] for baa in baa_set}
  gen_idx_baa_dict = {baa: [] for baa in baa_set}

  for idx, g in enumerate(generators):
    gen_baa_dict[g['BAA']].append(g)
    gen_idx_baa_dict[g['BAA']].append(idx)

  # group load by baa
  load_baa_dict = {baa: [] for baa in baa_set}

  for l in ctf_loads:
    load_baa_dict[l['BAA']].append(l)

  # useful groups
  non_ghg_baas = [g['BAA'] for g in generators if not g['ghg_area']]
  non_ghg_gen = [g for g in generators if g['BAA'] in non_ghg_baas]
  non_ghg_load = [l for l in ctf_loads if l['BAA'] in non_ghg_baas]
  ghg_gen = [g for g in generators if g['BAA'] not in non_ghg_baas]
  ghg_gen_idx = [idx for idx, g in enumerate(generators) if g['BAA'] not in non_ghg_baas]
  

  #vistra ctf
  vistra_ctf_dispatch = []

  for baa in baa_set:
    
    if gen_baa_dict[baa][0]['ghg_area']:
      vistra_ctf_dispatch += [0 for g in gen_baa_dict[baa]]
    
    else:
      vistra_ctf = run_solver(
        gen_baa_dict[baa],
        load_baa_dict[baa],
        include_ghg = False
        )
  
      vistra_ctf_solver, _ = vistra_ctf
      vistra_ctf_dispatch += [round(v.solution_value()) for v in vistra_ctf_solver.variables()]
    
  
  # caiso ctf
  
  caiso_ctf = run_solver(
    non_ghg_gen,
    non_ghg_load,
    include_ghg = False
    )
    
  caiso_ctf_solver, _ = caiso_ctf
  caiso_ctf_dispatch = [round(v.solution_value()) for v in caiso_ctf_solver.variables()]
  caiso_ctf_dispatch += [0 for g in ghg_gen]  


  # no ghg ctf
  
  noghg_ctf = run_solver(
    generators,
    ctf_loads,
    include_ghg = False
    )
  
  noghg_ctf_solver, _ = noghg_ctf
  noghg_ctf_dispatch = [round(v.solution_value()) for v in noghg_ctf_solver.variables()]
  
  ghg_area_load = sum([l['load'] for l in ctf_loads if l['ghg_area']])
  ghg_area_dispatch = sum([disp for idx, disp in enumerate(noghg_ctf_dispatch) if idx in ghg_gen_idx])
  noghg_net_import_ref = ghg_area_load - ghg_area_dispatch
  
  
  # set new attribution caps
  
  vistra_generators = []
  caiso_generators = []
  noghg_generators = []
  
  for idx, gen in enumerate(generators):
    
    cp_gen = deepcopy(gen)
    cp_gen['attrib_cap'] -= vistra_ctf_dispatch[idx] 
    vistra_generators.append(cp_gen)
    
    cp_gen = deepcopy(gen)
    cp_gen['attrib_cap'] -= caiso_ctf_dispatch[idx] 
    caiso_generators.append(cp_gen)
    
    cp_gen = deepcopy(gen)
    cp_gen['attrib_cap'] -= noghg_ctf_dispatch[idx] 
    noghg_generators.append(cp_gen)
    
  
  # market runs
  
  no_ctf_run = run_solver(generators, market_loads)
  no_ctf_solver, no_ctf_solution = no_ctf_run
  no_ctf_solution = add_settlements(generators, market_loads, no_ctf_solution)
    
    
  vistra_run = run_solver(vistra_generators, market_loads)
  vistra_solver, vistra_solution = vistra_run
  vistra_solution = add_settlements(generators, market_loads, vistra_solution)
  
  
  caiso_run = run_solver(caiso_generators, market_loads)
  caiso_solver, caiso_solution = caiso_run
  caiso_solution = add_settlements(generators, market_loads, caiso_solution)
  
    
  noghg_run = run_solver(noghg_generators, market_loads, net_import_reference_mw = noghg_net_import_ref)
  noghg_solver, noghg_solution = noghg_run
  noghg_solution = add_settlements(generators, market_loads, noghg_solution)
  
  
  if write_tables:
  
    # create tables
    make_ctf_table(
      generators,
      gen_idx_baa_dict,
      vistra_ctf_dispatch, 
      caiso_ctf_dispatch, 
      noghg_ctf_dispatch, 
      f"{outdir}/ctf_table.png"
      )
    
    
    # LMP Table
    
    make_lmp_table(vistra_solution, caiso_solution, noghg_solution)\
    .save(f"{outdir}/LMP_table.png")
    
    
    # Market Run Table
    
    make_market_table(
      generators,
      gen_idx_baa_dict,
      non_ghg_baas,
      ["-", "-" , "-", "-"], 
      no_ctf_solution, 
      f"{outdir}/no_ctf_market_run.png"
      )
    
    make_market_table(
      generators,
      gen_idx_baa_dict,
      non_ghg_baas,
      vistra_ctf_dispatch, 
      vistra_solution, 
      f"{outdir}/vistra_market_run.png"
      )
      
    make_market_table(
      generators,
      gen_idx_baa_dict,
      non_ghg_baas,caiso_ctf_dispatch, 
      caiso_solution, 
      f"{outdir}/caiso_market_run.png"
      )
      
    make_market_table(
      generators,
      gen_idx_baa_dict,
      non_ghg_baas,
      noghg_ctf_dispatch, 
      noghg_solution, 
      f"{outdir}/noghg_market_run.png"
      )
    
    
    #generator settlement
    
    for gen in generators:
      make_gen_settlement_table(gen, vistra_solution, caiso_solution, noghg_solution)\
      .save(f"{outdir}/{gen['name']}_settlement.png")
      
      
    # revenue distribution
    make_rev_dist_table(generators, vistra_solution, caiso_solution, noghg_solution, f"{outdir}/non_ghg_revenue_distribution.png")
    
    
    # load settlement
    
    make_load_settlement_table(loads, vistra_solution, caiso_solution, noghg_solution)\
    .save(f"{outdir}/load_settlement.png")
    
    
    # op settlement
    
    make_op_settlement_table(generators, market_loads, vistra_solution, caiso_solution, noghg_solution)\
    .save(f"{outdir}/op_settlement.png")
      
      
  return {
  'vistra_solution': vistra_solution, 
  'caiso_solution': caiso_solution,
  'noghg_solution': noghg_solution
  }
  

def make_generator(
  energy_bid,
  energy_cap,
  ghg_adder,
  ghg_area,
  name,
  BAA,
  attrib_cap = None
  ):
    
  if attrib_cap is None:
    attrib_cap = energy_cap
  
  return {
  "energy_bid": energy_bid,
  "energy_cap": energy_cap, 
  "ghg_adder": ghg_adder,
  "attrib_cap": attrib_cap, 
  "ghg_area": ghg_area,
  "name": name,
  'BAA': BAA
  }

def make_load(
  load,
  ghg_area,
  BAA,
  name
):
  
  return {
  'load': load,
  'ghg_area': ghg_area,
  'BAA': BAA,
  'name': name
  }
  
