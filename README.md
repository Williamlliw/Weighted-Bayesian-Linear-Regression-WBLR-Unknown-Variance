# Weighted-Bayesian-Linear-Regression-WBLR-Unknown-Variance
A tiny-but-complete Python implementation of **Weighted Bayesian Linear Regression** when the observation noise variance is **unknown** and endowed with an Inverse-Gamma prior.   Analytical posterior updates and Student-t predictive distributions are provided for both single-output and multi-output cases.
---

## What's inside

| File | Description |
|------|-------------|
| `bayes_module.py` | Core module<br>`wblr_fit` / `wblr_fit_multiout` – analytical posterior<br>`wblr_pred` – posterior predictive mean & credible/variance bands |
| `demo_wblr.py`    | Self-contained example: synthetic heteroscedastic data → fit → 99 % credible band visualization |

---

## Features

* **Weighted observations** – each sample can have an individual weight ∈ [0, 1]  
* **Analytical solution** – no MCMC, no optimization, deterministic O(d³)  
* **Unknown noise variance** – Gamma prior on the precision → Student-t predictive distribution  
* **Multi-output ready** – single call to `wblr_fit_multiout` for (N × n_y) target matrix  
* **Predictive uncertainty** – returns mean + credible intervals or full predictive covariance  

---

## Install

Only three common scientific-Python packages are required:

```bash
pip install numpy scipy matplotlib
