# Organ Transport Routing under Travel Time Uncertainty

A basic Python optimization project that analyzes organ transport routing decisions under uncertain travel times using stochastic programming concepts.

---

## Project Overview

This project uses Python and Pyomo to model an organ transport routing problem where travel times vary under different traffic or uncertainty scenarios.

The objective is to select the best hospital route that maximizes expected survival benefits while considering organ viability constraints and uncertain travel conditions.

The model evaluates routing decisions using:
- Recourse Problem (RP)
- Expected Value of Perfect Information (EVPI)
- Expected Value Solution (EEV)
- Value of the Stochastic Solution (VSS)

---

## Features

- Scenario-based stochastic optimization
- Hospital selection under uncertain travel times
- Expected survival analysis
- Comparison of deterministic vs stochastic solutions
- Visualization of optimization metrics using graphs

---

## Technologies Used

- Python
- Pyomo
- Matplotlib
- HiGHS Solver

---

## Problem Components

### Hospitals
The model evaluates multiple hospitals with different survival benefit values.

### Travel Scenarios
Different travel-time scenarios are considered:
- Very Low
- Low
- Medium
- High
- Extreme

Each scenario has an associated probability.

### Viability Constraints
The organ viability time limit is varied to analyze how routing decisions change under different survival windows.

---

## Optimization Metrics

### RP (Recourse Problem)
Optimal stochastic solution considering uncertainty.

### EEV (Expected Value Solution)
Deterministic solution based on average travel times.

### WS (Wait-and-See Solution)
Best possible solution assuming perfect future information.

### EVPI
Measures the value of having perfect information.

### VSS
Measures the benefit of using stochastic optimization instead of deterministic methods.

---

## Visualizations

The program generates:
- RP vs EEV vs WS comparison plot
- EVPI and VSS analysis plot

These graphs help compare deterministic and stochastic decision-making approaches.

---
