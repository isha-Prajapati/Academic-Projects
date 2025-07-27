from pyomo.environ import *

# Hospital names and survival values
hospitals = ['A', 'B', 'C', 'D', 'E', 'F']
survival = {
    'A': 15,
    'B': 18,
    'C': 22,
    'D': 19,
    'E': 25,
    'F': 17
}

# Travel scenarios with probabilities
scenarios = ['Very Low', 'Low', 'Medium', 'High', 'Extreme']
prob = {'Very Low': 0.1, 'Low': 0.2, 'Medium': 0.4, 'High': 0.2, 'Extreme': 0.1}

# Travel times (in hours) per hospital per scenario
travel_times = {
    'Very Low': {'A': 2, 'B': 3, 'C': 4, 'D': 3, 'E': 5, 'F': 2},
    'Low':      {'A': 3, 'B': 4, 'C': 5, 'D': 4, 'E': 6, 'F': 3},
    'Medium':   {'A': 4, 'B': 6, 'C': 7, 'D': 5, 'E': 8, 'F': 5},
    'High':     {'A': 6, 'B': 8, 'C': 9, 'D': 7, 'E': 10, 'F': 6},
    'Extreme':  {'A': 8, 'B': 9, 'C': 10, 'D': 9, 'E': 12, 'F': 8}
}

# Time viability limit (hours)
viability = 8


# Model
model = ConcreteModel()
model.H = Set(initialize=hospitals)
model.S = Set(initialize=scenarios)

model.x = Var(model.H, domain=Binary)
model.y = Var(model.H, model.S, domain=Binary)

# Objective: Maximize expected life-years saved
def obj_rule(m):
    return sum(prob[s] * survival[h] * m.y[h, s] for h in m.H for s in m.S)
model.obj = Objective(rule=obj_rule, sense=maximize)

# Only one hospital can be selected
model.select_one = Constraint(expr=sum(model.x[h] for h in model.H) == 1)

# Feasibility constraints
def link_x_y(m, h, s):
    return m.y[h, s] <= m.x[h]
model.link = Constraint(model.H, model.S, rule=link_x_y)

def time_limit(m, h, s):
    if travel_times[s][h] <= viability:
        return m.y[h, s] <= 1
    else:
        return m.y[h, s] == 0

model.timelimit = Constraint(model.H, model.S, rule=time_limit)

# Solve
SolverFactory('highs').solve(model)

# Results
print("Selected hospital:")
for h in model.H:
    if model.x[h]() == 1:
        print(f" → {h}")
        
expected_survival = value(model.obj)
print(f"\nExpected life-years saved (RP): {expected_survival:.2f}")

# EEV: use average travel time, pick feasible hospital with max survival
# Calculate average travel times
avg_times = {
    h: sum(prob[s] * travel_times[s][h] for s in scenarios)
    for h in hospitals
}

# Select hospital under average data (only if avg_time <= viability)
feasible_eev = {
    h: survival[h] for h in hospitals if avg_times[h] <= viability
}

if feasible_eev:
    best_h = max(feasible_eev.items(), key=lambda x: x[1])[0]
    print(f"[EEV] Selected: {best_h}, Life-years (deterministic): {survival[best_h]}")
    
    # Evaluate expected outcome using this decision under uncertainty
    EEV = 0
    for s in scenarios:
        if travel_times[s][best_h] <= viability:
            EEV += prob[s] * survival[best_h]
        else:
            EEV += 0  # organ arrived too late
else:
    best_h = None
    EEV = 0
    print("[EEV] No hospital has feasible average travel time.")


# WS: Best hospital for each scenario
WS = 0
for s in scenarios:
    feasibles = {h: survival[h] for h in hospitals if travel_times[s][h] <= viability}
    if feasibles:
        best = max(feasibles.values())
    else:
        best = 0
    WS += prob[s] * best
print(f"[WS] Expected Life-years: {WS:.2f}")

# EVPI and VSS
RP = expected_survival
EVPI = WS - RP
VSS = RP - EEV
print(f"\n EVPI = {EVPI:.2f}")
print(f" VSS = {VSS:.2f}")
