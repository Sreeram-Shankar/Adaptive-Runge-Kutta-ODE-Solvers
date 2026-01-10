# Adaptive-Runge-Kutta-ODE-Solvers

A reference library implementing **adaptive step-size explicit and implicit Runge–Kutta ODE integrators**, using a **universal, method-agnostic step size controller based on an embedded Euler defect**.

This project is intentionally explicit and theory-driven.  
Adaptivity is implemented *orthogonally* to the numerical method itself, without classical embedded pairs, local truncation error estimators, or stiffness heuristics.

The goal is **clarity, correctness, and experimental freedom**, not production performance.

---

## Scope and Purpose

This library explores a specific idea:

> **Adaptive step sizing can be driven by geometric failure of first-order extrapolation (Euler defect), rather than method-specific LTE estimators.**

As a result:

- All methods share the **same step size controller**
- Accuracy comes from the **method**, not the controller
- Stability comes from **implicitness**, not heuristics

This makes the library ideal for:
- solver design experiments
- benchmarking explicit vs implicit behavior
- studying stiffness, curvature, and nonlinearity independently

---

## Features

### Adaptive Explicit Runge–Kutta

- RK2 through RK7
- Fixed Butcher tableaus
- Adaptive step size via **Euler defect PI controller**
- No embedded RK pairs
- No order-dependent error scaling
- Python and Julia implementations

Intended for:
- nonstiff problems
- curvature-dominated dynamics
- baselines and comparisons

---

### Adaptive Implicit Runge–Kutta (Collocation)

- **Gauss–Legendre** methods  
- **Radau IIA** methods  
- **Lobatto IIIC** methods  
- Arbitrary stage count (practical limits apply)

Features:
- Newton iteration per step
- Rejection on Newton failure
- Rejection on catastrophic Euler defect
- Same adaptive controller as explicit RK

Implicitness provides:
- stiffness robustness
- stability, not larger steps by default

---

### Unified Adaptive Step Size Controller

All solvers use the same controller:

- **Embedded Euler predictor**
- Defect:
  \[
  d = y_{n+1}^{(method)} - (y_n + h f_n)
  \]
- Scaled via WRMS norm with `atol` / `rtol`
- PI-controlled step updates

Important properties:

- Method-agnostic
- Order-honest (treated as first-order defect)
- Geometric, not asymptotic
- Independent of stiffness detection

---

## What This Library Is *Not*

- ❌ No classical LTE estimators  
- ❌ No embedded RK pairs (Dormand–Prince, etc.)  
- ❌ No adaptive multistep methods  
- ❌ No Jacobian reuse or performance tuning  
- ❌ Not a replacement for CVODE, Sundials, or DifferentialEquations.jl  

This library prioritizes **conceptual transparency over efficiency**.

---

## Design Philosophy

### Single-Step Methods Only

Only one-step methods are included:

- Explicit Runge–Kutta  
- Implicit Runge–Kutta (collocation)

This ensures:
- clean rejection semantics
- no step history interpolation
- no hidden state coupling step size and history

---

### Orthogonal Control Axes

Adaptivity is decomposed into two independent mechanisms:

| Aspect                  | Controlled by                  |
|-------------------------|--------------------------------|
| Geometry / curvature    | Euler defect                   |
| Stiffness / stability   | Implicit Newton convergence    |

This separation is intentional and central to the design.

---

### Accuracy Comes From the Method

The controller:
- decides *whether* a step is acceptable
- does **not** attempt to estimate the method’s formal LTE

As a result:
- higher-order methods are more accurate
- but do not automatically take larger steps
- step size reflects solution geometry, not order

---

## Example Usage

### Adaptive Explicit RK

```python
from rk import solve_rk4_adaptive

def f(t, y):
    return -y

T, Y, E = solve_rk4_adaptive(
    f,
    t_span=(0.0, 5.0),
    y0=[1.0],
    h0=0.1,
    atol=1e-6,
    rtol=1e-3
)
```
Adaptive Implicit Runge–Kutta (Collocation)
```python
from irk import solve_collocation_adaptive

def f(t, y):
    return -y

T, Y, E = solve_collocation_adaptive(
    f,
    t_span=(0.0, 5.0),
    y0=[1.0],
    h0=0.1,
    family="gauss",
    s=3,
    atol=1e-6,
    rtol=1e-3
)
```
---

## Notes on Behavior and Interpretation
Because step size is governed by Euler defect:

- Explicit and implicit methods often take similar step sizes on smooth problems

- Implicit methods shine on stiff transients, not smooth curvature

- Higher-order RK improves accuracy within accepted steps

- Stiffness is handled through Newton success, not detection heuristics

These behaviors are expected and reflect the library’s design.
