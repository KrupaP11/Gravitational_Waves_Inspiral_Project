# ML Regression on Synthetic Gravitational-Wave Inspirals

A machine learning surrogate model for gravitational-wave binary inspirals, built as part of a course project. A synthetic dataset of 1,000 samples is generated from first-principles physics, and a Random Forest regression model is trained to recover binary parameters from strain time-series features. For full derivations and methodology, see the [project report](report/main.tex).

---

## Overview

When two massive objects (such as black holes or neutron stars) orbit each other, they emit gravitational waves and slowly spiral inward — a process called an **inspiral**. The key parameters describing this system are the **chirp mass** (which controls how fast the binary spirals inward) and the **merger time** (how long until the two objects collide).

This project investigates whether a Random Forest can recover these parameters from simulated gravitational-wave strain data. The short answer: it can, but not for the right reasons.

---

## Physics

The dataset is generated using:

- **Peters (1964) orbital decay formula** — describes how the orbital separation shrinks over time due to gravitational-wave emission
- **Quadrupole strain approximation** — gives the gravitational-wave strain amplitude $h(t)$ as a function of chirp mass, orbital frequency, and observer distance
- **Kepler's law** — used to derive the orbital frequency from the separation

Numerical integration of the orbital decay ODE is performed with `scipy.integrate.solve_ivp` (RK45).

---

## Dataset

| Property | Value |
|---|---|
| Samples | 1,000 |
| Features | 20 |
| Targets | 2 |

**Features:**
- Noisy strain values at 17 time stamps $h_\text{noisy}(t=1,\dots,17)$
- Distance to observer $d$
- Initial orbital separation $r_0$
- Reduced mass $\mu = m_1 m_2 / (m_1 + m_2)$

**Targets:**
- Chirp mass $\mathcal{M}$
- Time to merger $t_m$

**Parameter ranges:**

| Parameter | Range |
|---|---|
| $m_1, m_2$ | $1.2 - 50\ M_\odot$ (neutron stars to stellar-mass black holes) |
| $r_0$ | $5 \times 10^6$ km to $1 \times 10^8$ km |
| $d$ | $10$ kpc to $10^5$ kpc |

---

## Model

A **Random Forest** regressor (`scikit-learn`) was chosen for its ability to handle nonlinear relationships and mixed-scale features without polynomial feature engineering.

| Metric | Train | Test |
|---|---|---|
| $R^2$ | 0.87 | 0.59 |
| MAE | $4.14 \times 10^{29}$ | $1.08 \times 10^{30}$ |

---

## Key Finding

Feature importance analysis revealed that the model was **not** learning from the strain signal. Instead, it exploited the fact that $r_0$ and $\mu$ appear directly in the analytical formulas for merger time and chirp mass — effectively reconstructing an algebraic shortcut rather than learning any physics.

When restricted to strain-only features, the model failed entirely, defaulting to predicting the mean value. This highlights why real gravitational-wave parameter estimation (as done at LIGO) relies on matched filtering and Bayesian inference rather than simple regression on raw strain.

---

## Requirements

```
numpy
scipy
scikit-learn
matplotlib
```

Install with:
```bash
pip install numpy scipy scikit-learn matplotlib
```

---

## Usage

```bash
python generate_data.py      # Generate synthetic inspiral dataset
python train_model.py        # Train Random Forest and evaluate
python plot_results.py       # Generate predicted vs. actual and feature importance plots
```

---

## References

- Peters, P. C. (1964). *Gravitational radiation and the motion of two point masses.* Physical Review, 136(4B).
- Creighton, T. Formulae and details: Gravitational wave details. Caltech TAPIR. https://www.tapir.caltech.edu/~teviet/Waves/gwave_details.html#inspiral
- Veitch, J. et al. (2015). *Parameter estimation for compact binaries with LALInference.* Physical Review D, 91.
- Ashton, G. et al. (2019). *Bilby: A user-friendly Bayesian inference library for gravitational-wave astronomy.* ApJS, 241.
