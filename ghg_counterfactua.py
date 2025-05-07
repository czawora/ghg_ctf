
from scipy.optimize import linprog
from great_tables import GT
import pandas as pd


gen1_cap = 200
gen2_cap = 95
gen3_cap = 140 
gen4_cap = 100


def run_opt(ghg_load = None, bounds = None):
  
  if bounds is None:
    bounds = [
      (0, gen1_cap),
      (0, gen1_cap),
      (0, gen2_cap),
      (0, gen2_cap),
      (0, gen3_cap),
      (0, gen3_cap),
      (0, gen4_cap),
      (0, gen4_cap)
      ]

  if ghg_load is None:
    ghg_load = 100
  total_load = 230
  
  cost = [
    [8, 13],
    [10, 11],
    [2, 2],
    [100, 110]
    ]
  cost = [item for row in cost for item in row]
  
  lhs_ineq_constraint = [
    [1, 1] + ([0] * 6), 
    ([0] * 2) + [1, 1] + ([0] * 4),
    ([0] * 4) + [1, 1] + ([0] * 2),
    ([0] * 6) + [1, 1]
    ]
  rhs_ineq_constraint = [gen1_cap, gen2_cap, gen3_cap, gen4_cap]
  
  lhs_eq_constraint = [  
    [0,1] * 4,
    [1] * 8
    ]
  rhs_eq_constraint = [
    ghg_load, 
    total_load
    ]

  res = linprog(
    cost, 
    A_ub=lhs_ineq_constraint, 
    b_ub=rhs_ineq_constraint, 
    A_eq = lhs_eq_constraint,
    b_eq = rhs_eq_constraint,
    bounds=bounds
    )
    
  return res


# CAISO counterfactual
def run_CAISO():
  
  total_load = 130
  
  bounds = [
      (0, gen1_cap),
      (0, gen1_cap),
      (0, gen2_cap),
      (0, gen2_cap),
      (0, gen3_cap),
      (0, gen3_cap)
      ]
  
  cost = [
    [8, 13],
    [10, 11],
    [2, 2]
    ]
  cost = [item for row in cost for item in row]
  
  lhs_ineq_constraint = [
    [1, 1] + ([0] * 4), 
    ([0] * 2) + [1, 1] + ([0] * 2),
    ([0] * 4) + [1, 1]
    ]
  rhs_ineq_constraint = [gen1_cap, gen2_cap, gen3_cap]
  
  lhs_eq_constraint = [  
    [1] * 6
    ]
  rhs_eq_constraint = [total_load]
  
  res = linprog(
    cost, 
    A_ub=lhs_ineq_constraint, 
    b_ub=rhs_ineq_constraint, 
    A_eq = lhs_eq_constraint,
    b_eq = rhs_eq_constraint,
    bounds=bounds
    )
    
  return res

CAISO_ctf_res = run_CAISO()
NOGHG_ctf_res = run_opt(ghg_load = 0)

# manually set vistra outcomes

CAISO_ctf_dispatch = CAISO_ctf_res.x.tolist() + [0, 0]
NOGHG_ctf_dispatch = NOGHG_ctf_res.x.tolist()
VISTRA_ctf_dispatch = [100, 0, 0, 0, 30, 0, 0, 0] # keep last item as 0, even though dispatch was 100

# set new bounds

ubs = [gen1_cap, gen1_cap, gen2_cap, gen2_cap, gen3_cap, gen3_cap, gen4_cap, gen4_cap]

CAISO_bounds = []
NOGHG_bounds = []
VISTRA_bounds = []

for idx, i in enumerate(ubs):
  
  if idx % 2 == 1:
    CAISO_bounds.append((0, i - CAISO_ctf_dispatch[idx - 1]))
    NOGHG_bounds.append((0, i - NOGHG_ctf_dispatch[idx - 1]))
    VISTRA_bounds.append((0, i - VISTRA_ctf_dispatch[idx - 1]))
  else:
    CAISO_bounds.append((0, i))
    NOGHG_bounds.append((0, i))
    VISTRA_bounds.append((0, i))
    
# run with new bounds

CAISO_res = run_opt(bounds = CAISO_bounds)
NOGHG_res = run_opt(bounds = NOGHG_bounds)
VISTRA_res = run_opt(bounds = VISTRA_bounds)

CAISO_attribution = [i for idx, i in enumerate(CAISO_res.x) if idx % 2 == 1]
NOGHG_attribution = [i for idx, i in enumerate(NOGHG_res.x) if idx % 2 == 1]
VISTRA_attribution = [i for idx, i in enumerate(VISTRA_res.x) if idx % 2 == 1]

table_data = {
  "": ["SMEC", "MC-GHG", "Gen 1 Attr", "Gen 2 Attr", "Gen 3 Attr", "Gen 4 Attr"],
  "CAISO": [CAISO_res.eqlin.marginals[1], CAISO_res.eqlin.marginals[0]] + CAISO_attribution,
  "VISTRA": [VISTRA_res.eqlin.marginals[1], VISTRA_res.eqlin.marginals[0]] + VISTRA_attribution,
  "No GHG Cost": [NOGHG_res.eqlin.marginals[1], NOGHG_res.eqlin.marginals[0]] + NOGHG_attribution
  }

GT(pd.DataFrame(table_data)).save("/Users/chriszawora/Dropbox/WIEB/linprog/table.png")



L1 = 100
L2 = 30
L3 = 100


def run_opt2(bounds = None):
  
  if bounds is None:
    bounds = [
      (0, gen1_cap),
      (0, None),
      (0, gen2_cap),
      (0, None),
      (0, gen3_cap),
      (0, None),
      (0, gen4_cap),
      (0, None)
      ]

  total_load = sum([L1, L2, L3])
  
  cost = [8, 5, 10, 1, 2, 0, 100, 10]

  lhs_ineq_constraint = [
    [-1, 1] + ([0] * 6),
    ([0] * 2) + [-1, 1] + ([0] * 4),
    ([0] * 4) + [-1, 1] + ([0] * 2),
    ([0] * 6) + [-1, 1]
    ]
  rhs_ineq_constraint = [0, 0, 0, 0]
  
  lhs_eq_constraint = [  
    [1, -1] * 3 + ([0] * 2), 
    [1, 0] * 4
    ]
    
  rhs_eq_constraint = [
    L1 + L2,
    total_load
    ]

  res = linprog(
    cost, 
    A_ub=lhs_ineq_constraint, 
    b_ub=rhs_ineq_constraint, 
    A_eq = lhs_eq_constraint,
    b_eq = rhs_eq_constraint,
    bounds=bounds
    )
    
  return res


def run_NOGHG(bounds = None):
  
  if bounds is None:
    bounds = [
      (0, gen1_cap),
      (0, gen2_cap),
      (0, gen3_cap),
      (0, gen4_cap)
      ]

  total_load = sum([L1, L2, L3])
  
  cost = [8, 10, 2, 100]
  
  lhs_eq_constraint = [  
    [1] * 4
    ]
    
  rhs_eq_constraint = [
    total_load
    ]

  res = linprog(
    cost, 
    A_eq = lhs_eq_constraint,
    b_eq = rhs_eq_constraint,
    bounds=bounds
    )
    
  return res


# CAISO counterfactual
def run_CAISO2():
  
  total_load = 130
  
  bounds = [
      (0, gen1_cap),
      (0, 0),
      (0, gen2_cap),
      (0, 0),
      (0, gen3_cap),
      (0, 0)
      ]
  
  cost = [
    [8, 13],
    [10, 11],
    [2, 2]
    ]
  cost = [item for row in cost for item in row]
  
  lhs_ineq_constraint = [
    [1, 1] + ([0] * 4), 
    ([0] * 2) + [1, 1] + ([0] * 2),
    ([0] * 4) + [1, 1]
    ]
  rhs_ineq_constraint = [gen1_cap, gen2_cap, gen3_cap]
  
  lhs_eq_constraint = [  
    [1] * 6
    ]
  rhs_eq_constraint = [total_load]
  
  res = linprog(
    cost, 
    A_ub=lhs_ineq_constraint, 
    b_ub=rhs_ineq_constraint, 
    A_eq = lhs_eq_constraint,
    b_eq = rhs_eq_constraint,
    bounds=bounds
    )
    
  return res


CAISO_ctf_res2 = run_CAISO2()
NOGHG_ctf_res2 = run_NOGHG()

# manually set vistra outcomes

CAISO_ctf_dispatch2 = CAISO_ctf_res2.x.tolist() + [0, 0]
NOGHG_ctf_dispatch2 = [90, 0, 0, 0, 140, 0, 0, 0]#NOGHG_ctf_res2.x.tolist()
VISTRA_ctf_dispatch2 = [100, 0, 0, 0, 30, 0, 0, 0] # keep last item as 0, even though dispatch was 100

# set new bounds

ubs2 = [gen1_cap, gen1_cap, gen2_cap, gen2_cap, gen3_cap, gen3_cap, gen4_cap, gen4_cap]

CAISO_bounds2 = []
NOGHG_bounds2 = []
VISTRA_bounds2 = []

for idx, i in enumerate(ubs2):
  
  if idx % 2 == 1:
    CAISO_bounds2.append((0, i - CAISO_ctf_dispatch2[idx - 1]))
    NOGHG_bounds2.append((0, i - NOGHG_ctf_dispatch2[idx - 1]))
    VISTRA_bounds2.append((0, i - VISTRA_ctf_dispatch2[idx - 1]))
  else:
    CAISO_bounds2.append((0, i))
    NOGHG_bounds2.append((0, i))
    VISTRA_bounds2.append((0, i))
    
# run with new bounds

CAISO_res2 = run_opt2(bounds = CAISO_bounds2)
NOGHG_res2 = run_opt2(bounds = NOGHG_bounds2)
VISTRA_res2 = run_opt2(bounds = VISTRA_bounds2)

CAISO_attribution2 = [i for idx, i in enumerate(CAISO_res2.x) if idx % 2 == 1]
NOGHG_attribution2 = [i for idx, i in enumerate(NOGHG_res2.x) if idx % 2 == 1]
VISTRA_attribution2 = [i for idx, i in enumerate(VISTRA_res2.x) if idx % 2 == 1]

table_data2 = {
  "": ["SMEC", "MC-GHG", "Gen 1 Attr", "Gen 2 Attr", "Gen 3 Attr", "Gen 4 Attr"],
  "CAISO": [CAISO_res2.eqlin.marginals[1], CAISO_res2.eqlin.marginals[0]] + CAISO_attribution2,
  "VISTRA": [VISTRA_res2.eqlin.marginals[1], VISTRA_res2.eqlin.marginals[0]] + VISTRA_attribution2,
  "No GHG Cost": [NOGHG_res2.eqlin.marginals[1], NOGHG_res2.eqlin.marginals[0]] + NOGHG_attribution2
  }

GT(pd.DataFrame(table_data2)).save("/Users/chriszawora/Dropbox/WIEB/linprog/table2.png")


gen1_cap = 200
gen2_cap = 100
gen3_cap = 140 
gen4_cap = 100


L1 = 100
L2 = 30
L3 = 100


# OR TOOLS

from ortools.linear_solver import pywraplp

#get solver 
solver = pywraplp.Solver.CreateSolver('GLOP')

# declare decision variables
gen1 = solver.NumVar(0.0, gen1_cap, 'gen1')
gen2 = solver.NumVar(0.0, gen2_cap, 'gen2')
gen3 = solver.NumVar(0.0, gen3_cap, 'gen3')
gen4 = solver.NumVar(0.0, gen4_cap, 'gen4')

gen1_attrib = solver.NumVar(0.0, 100, 'gen1_attrib')
gen2_attrib = solver.NumVar(0.0, gen2_cap, 'gen2_attrib')
gen3_attrib = solver.NumVar(0.0, 0, 'gen3_attrib')
gen4_attrib = solver.NumVar(0.0, gen4_cap, 'gen4_attrib')


# declare objective
solver.Minimize(
  gen1*8 + gen1_attrib*5 + gen2*10 + gen2_attrib*1 + gen3*2 + gen3_attrib*0 + gen4*100 + gen4_attrib*10)


# declare constraints
constraints = {
  "c1": solver.Add(gen1_attrib - gen1 <= 0).name(),
  "c2": solver.Add(gen2_attrib - gen2 <= 0).name(),
  "c3": solver.Add(gen3_attrib - gen3 <= 0).name(),
  "c4": solver.Add(gen4_attrib - gen4 <= 0).name(),
  "c5": solver.Add(gen1 + gen2 + gen3 - gen1_attrib - gen2_attrib - gen3_attrib <= L1 + L2).name(),
  "c6": solver.Add(gen1 + gen2 + gen3 + gen4 == L1 + L2 + L3).name()
}
constraints = {v: k for k, v in constraints.items()}


# solve
results = solver.Solve()

# print results
if results == pywraplp.Solver.OPTIMAL: print(f'The solution is optimal.') 

print(f'Objective value: z* = {solver.Objective().Value()}')

for i in solver.variables():
  print(f'Name: {i.name()}, Optimal value: {i.solution_value()}')
  
for i in solver.constraints():
  print(f'Name: {constraints[i.name()]}, Shadow Price: {i.dual_value()}')
  
  
from great_tables import GT, md, html, google_font, style
  
def run_solver(
  ghg_area_load,
  non_ghg_area_load,
  gen_energy_costs, 
  gen_energy_caps, 
  gen_ghg_adder = None, 
  gen_attrib_caps = None
  ):
  
  total_load = ghg_area_load + non_ghg_area_load
  
  #get solver 
  solver = pywraplp.Solver.CreateSolver('GLOP')
  
  # declare decision variables
  gen_energy_vars = []
  for idx, cap in enumerate(gen_energy_caps):
    gen_energy_vars.append(solver.NumVar(0.0, gen_energy_caps[idx], f'gen{idx + 1}_energy'))
    
  obj_func = sum([bid * gen for bid, gen in zip(gen_energy_costs, gen_energy_vars)])
    
  if gen_attrib_caps is not None:
    gen_attrib_vars = []
    for idx, cap in enumerate(gen_attrib_caps):
      gen_attrib_vars.append(solver.NumVar(0.0, gen_attrib_caps[idx], f'gen{idx + 1}_attrib'))

    obj_func += sum([bid * gen for bid, gen in zip(gen_ghg_adder, gen_attrib_vars)])
  
  solver.Minimize(obj_func)
  
  # declare constraints
  constraints = {}
  
  constraints.update({
    solver.Add(sum(gen_energy_vars) == total_load).name(): "power_balance_constraint"
  })
  
  if gen_attrib_caps is not None:
    for idx, vars in enumerate(zip(gen_energy_vars, gen_attrib_vars)):
      constraints.update({solver.Add(vars[1] - vars[0] <= 0).name(): f"alloc_limit{idx + 1}"})
    
    constraints.update({
      solver.Add(
        sum(gen_energy_vars[0:3]) + (-1) * sum(gen_attrib_vars[0:3]) <= non_ghg_area_load).name(): 
        "export_constraint"
      })
      
  results = solver.Solve()
  
  if results != pywraplp.Solver.OPTIMAL:
    raise("Solution Not Optimal")
  
  solution = {}
  
  for var in solver.variables():
    solution.update({var.name(): var.solution_value()})
    
  for ct in solver.constraints():
    solution.update({constraints[ct.name()]: ct.dual_value()})
  
  return solver, solution



L1 = 100
L2 = 30
L3 = 100

gen1 = {
  "energy_bid": 8,
  "energy_cap": 200,
  "ghg_adder": 5,
  "attrib_cap": 200 
}

gen2 = {
  "energy_bid": 10,
  "energy_cap": 95,
  "ghg_adder": 1,
  "attrib_cap": 95 
}

gen3 = {
  "energy_bid": 2,
  "energy_cap": 140,
  "ghg_adder": 0,
  "attrib_cap": 140
}

gen4 = {
  "energy_bid": 100,
  "energy_cap": 100,
  "ghg_adder": 10,
  "attrib_cap": 100 
}


gen_energy_costs = []
gen_energy_caps = []
gen_ghg_adder = []
gen_attrib_caps = []


for gen in [gen1, gen2, gen3, gen4]:
  gen_energy_costs.append(gen['energy_bid'])
  gen_energy_caps.append(gen['energy_cap'])
  gen_ghg_adder.append(gen['ghg_adder'])
  gen_attrib_caps.append(gen['attrib_cap'])

# run counterfactuals

vistra_ctf_dispatch = [100, 0, 30, 0]


caiso_ctf = run_solver(
  0,
  L1 + L2,
  gen_energy_costs[0:3], 
  gen_energy_caps[0:3], 
  None, 
  None
  )
  
caiso_ctf_solver, _ = caiso_ctf
caiso_ctf_dispatch = [round(v.solution_value()) for v in caiso_ctf_solver.variables()][0:3] + [0]


noghg_ctf = run_solver(
  0,
  L1 + L2 + L3,
  gen_energy_costs, 
  gen_energy_caps, 
  None, 
  None
  )

noghg_ctf_solver, _ = noghg_ctf
noghg_ctf_dispatch = [round(v.solution_value()) for v in noghg_ctf_solver.variables()]


# set new attribution caps

vistra_gen_attrib_caps = [cap - ctf for cap, ctf in zip(gen_attrib_caps, vistra_ctf_dispatch)]
caiso_gen_attrib_caps = [cap - ctf for cap, ctf in zip(gen_attrib_caps, caiso_ctf_dispatch)]
noghg_gen_attrib_caps = [cap - ctf for cap, ctf in zip(gen_attrib_caps, noghg_ctf_dispatch)]


# market runs

vistra_run = run_solver(
  L3,
  L1 + L2,
  gen_energy_costs, 
  gen_energy_caps, 
  gen_ghg_adder, 
  vistra_gen_attrib_caps
  )

vistra_solver, vistra_solution = vistra_run


caiso_run = run_solver(
  L3,
  L1 + L2,
  gen_energy_costs, 
  gen_energy_caps, 
  gen_ghg_adder, 
  caiso_gen_attrib_caps
  )

caiso_solver, caiso_solution = caiso_run

  
noghg_run = run_solver(
  L3,
  L1 + L2,
  gen_energy_costs, 
  gen_energy_caps, 
  gen_ghg_adder, 
  noghg_gen_attrib_caps
  )

noghg_solver, noghg_solution = noghg_run


# create counterfactual table

ctf_table = {
  "": ["Vistra", "CAISO", "No GHG Cost"],
  "BAA A": [
    f'G1: {vistra_ctf_dispatch[0]} MW', 
    f'G1: {caiso_ctf_dispatch[0]} MW', 
    f'G1: {noghg_ctf_dispatch[0]} MW'
  ],
  "BAA B": [
    f'G2: {vistra_ctf_dispatch[1]} MW\rG3: {vistra_ctf_dispatch[2]} MW', 
    f'G2: {caiso_ctf_dispatch[1]} MW\rG3: {caiso_ctf_dispatch[2]} MW', 
    f'G2: {noghg_ctf_dispatch[1]} MW\rG3: {noghg_ctf_dispatch[2]} MW'
  ],
  "BAA C": [
    f'G4: {vistra_ctf_dispatch[3]} MW', 
    f'G4: {caiso_ctf_dispatch[3]} MW', 
    f'G4: {noghg_ctf_dispatch[3]} MW'
  ],
  }

GT(pd.DataFrame(ctf_table))\
.tab_header(
      title=md("**Counterfactual Dispatch**")
  )\
.tab_options(
    style=style.text(font=google_font("Amasis MT Pro")),
    # data_row_padding='1px',
    # heading_background_color='antiquewhite',
    # source_notes_background_color='antiquewhite',
    # column_labels_background_color='antiquewhite',
    # table_background_color='snow',
    # data_row_padding_horizontal=3,
    # column_labels_padding_horizontal=58
    ) \
.cols_align(    
    align='center'
    ) \
.save("/Users/chriszawora/Dropbox/WIEB/linprog/example1/ctf_table.png")

