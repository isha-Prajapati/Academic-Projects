from pyomo.environ import *
import matplotlib.pyplot as plt
import csv

# Hospital names and survival values
hospitals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
survival = {
    'A': 15, 'B': 18, 'C': 22, 'D': 19, 'E': 25, 'F': 17,
    'G': 20, 'H': 16, 'I': 23, 'J': 21
}


# Travel scenarios and probabilities
scenarios = ['Very Low', 'Low', 'Medium', 'High', 'Extreme']
prob = {'Very Low': 0.1, 'Low': 0.2, 'Medium': 0.4, 'High': 0.2, 'Extreme': 0.1}

# Travel times
travel_times = {
    'Very Low': {'A': 2, 'B': 3, 'C': 4, 'D': 3, 'E': 5, 'F': 2, 'G': 4, 'H': 3, 'I': 5, 'J': 4},
    'Low':      {'A': 3, 'B': 4, 'C': 5, 'D': 4, 'E': 6, 'F': 3, 'G': 5, 'H': 4, 'I': 6, 'J': 5},
    'Medium':   {'A': 4, 'B': 6, 'C': 7, 'D': 5, 'E': 8, 'F': 5, 'G': 7, 'H': 6, 'I': 8, 'J': 7},
    'High':     {'A': 6, 'B': 8, 'C': 9, 'D': 7, 'E': 10, 'F': 6, 'G': 9, 'H': 8, 'I': 10, 'J': 9},
    'Extreme':  {'A': 8, 'B': 9, 'C': 10, 'D': 9, 'E': 12, 'F': 8, 'G': 11, 'H': 10, 'I': 12, 'J': 11}
}

# Viability values to test
viability_range = list(range(6, 13))

# Output storage
results = []

for viability in viability_range:
    # Pyomo model
    model = ConcreteModel()
    model.H = Set(initialize=hospitals)
    model.S = Set(initialize=scenarios)

    model.x = Var(model.H, domain=Binary)
    model.y = Var(model.H, model.S, domain=Binary)

    def obj_rule(m):
        return sum(prob[s] * survival[h] * m.y[h, s] for h in m.H for s in m.S)
    model.obj = Objective(rule=obj_rule, sense=maximize)

    model.select_one = Constraint(expr=sum(model.x[h] for h in model.H) == 1)

    def link_x_y(m, h, s):
        return m.y[h, s] <= m.x[h]
    model.link = Constraint(model.H, model.S, rule=link_x_y)

    def time_limit(m, h, s):
        if travel_times[s][h] <= viability:
            return m.y[h, s] <= 1
        else:
            return m.y[h, s] == 0
    model.timelimit = Constraint(model.H, model.S, rule=time_limit)

    SolverFactory('highs').solve(model, tee=False)
    RP = value(model.obj)

    selected_hospital = None
    for h in hospitals:
        if value(model.x[h]) > 0.5:  # since x[h] is binary, value close to 1 means selected
            selected_hospital = h
            break

    # EEV
    avg_times = {h: sum(prob[s] * travel_times[s][h] for s in scenarios) for h in hospitals}
    feasible_eev = {h: survival[h] for h in hospitals if avg_times[h] <= viability}
    if feasible_eev:
        best_h = max(feasible_eev.items(), key=lambda x: x[1])[0]
        EEV = sum(prob[s] * survival[best_h] if travel_times[s][best_h] <= viability else 0 for s in scenarios)
    else:
        EEV = 0

    # WS
    WS = 0
    for s in scenarios:
        feasibles = {h: survival[h] for h in hospitals if travel_times[s][h] <= viability}
        WS += prob[s] * (max(feasibles.values()) if feasibles else 0)

    # EVPI and VSS
    EVPI = WS - RP
    VSS = RP - EEV

  

    results.append((viability, RP, EEV, WS, EVPI, VSS, selected_hospital))




# === Plot RP, EEV, WS ===
viab, rp_vals, eev_vals, ws_vals, evpi_vals, vss_vals, hosp_vals = zip(*results)
plt.figure(figsize=(10,6))
plt.plot(viab, rp_vals, label='RP (Stochastic)', marker='o')
plt.plot(viab, eev_vals, label='EEV (Deterministic)', marker='s')
plt.plot(viab, ws_vals, label='WS (Perfect Info)', marker='^')
plt.xlabel('Viability Limit (hours)')
plt.ylabel('Expected Life-Years Saved')
plt.title('RP, EEV, and WS vs. Viability Limit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot EVPI and VSS ===
plt.figure(figsize=(10,6))
plt.plot(viab, evpi_vals, label='EVPI', color='red', marker='x')
plt.plot(viab, vss_vals, label='VSS', color='purple', marker='d')
plt.xlabel('Viability Limit (hours)')
plt.ylabel('Value (Life-Years)')
plt.title('EVPI and VSS vs. Viability Limit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print Table ===
print(f"{'Viability':>10} | {'Hospital':>8} | {'RP':>6} | {'EEV':>6} | {'WS':>6} | {'EVPI':>6} | {'VSS':>6}")
print("-" * 65)
for row in results:
    print(f"{row[0]:>10} | {row[6]:>8} | {row[1]:6.2f} | {row[2]:6.2f} | {row[3]:6.2f} | {row[4]:6.2f} | {row[5]:6.2f}")


