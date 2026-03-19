# SRPF — Spectral Resonance Pre-Filtering for Ballistic Missile Defence

> **A spectral-geometric approach to rapid threat discrimination under saturation attack scenarios**

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](./paper/XJR_SRPF_Missile_Defence.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-yellow.svg)](https://python.org)

---

## Overview

Modern short-range missile defence systems (Iron Dome, David's Sling, and similar architectures) face a critical computational bottleneck during **saturation attacks**: when dozens or hundreds of projectiles are launched simultaneously, the number of radar returns can overwhelm the tracking pipeline, forcing the system to either delay response or classify targets with incomplete information.

**SRPF** introduces a lightweight **spectral pre-filtering layer** that operates *upstream* of the Kalman filter, exploiting the fact that ballistic trajectories produce radar innovation sequences with a characteristic spectral structure that is absent in clutter, debris, and decoys.

The core insight — that **physical or arithmetic structure imposes spectral regularity detectable via intrinsic reference frequencies** — is a direct transposition of the [Spectral Sieve Pre-filtering (SSP)](https://doi.org/10.5281/zenodo.19002607) framework from number-theoretic cryptanalysis to the kinematic domain.

## Key Results

| Metric | Value | Reference |
|--------|-------|-----------|
| ROC AUC (SRPF vs Energy Detection) | **0.969** vs 0.625 | Theorem 5.4 |
| Computational gain factor G(τ*) | **×2.75** | Theorem 4.3 |
| Filter rate at P_D = 90% | **67%** of tracks discarded | Section 8.5 |
| Saturation scenario improvement | **+91%** threats detected | Theorem 6.1 |
| PSD level ratio (clutter / ballistic) | **65×** | Theorems 3.1–3.2 |

## How It Works

```
Radar returns (M tracks)
        │
        ▼
┌───────────────────┐
│  Track Initiation  │   m-of-n detection logic
│  (N measurements)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   SRPF Pre-Filter  │   O(N·J) per track — lightweight
│   σ(Z_N) ≥ τ ?    │   ballistic resonance score
└──┬─────────┬──────┘
   │ YES     │ NO
   ▼         ▼
 ┌─────┐  ┌──────┐
 │ EKF │  │ Drop │   non-ballistic, saves O(N·n³)
 └──┬──┘  └──────┘
    │
    ▼
 Threat Assessment → Intercept
```

The **ballistic resonance score** σ combines four features extracted from the detrended innovation residuals:

1. **Normalised innovation energy** — ballistic targets produce small residuals (good model fit)
2. **Whiteness test (x-axis)** — ballistic residuals are nearly uncorrelated
3. **Whiteness test (y-axis)** — same for the vertical component  
4. **Spectral flatness** — Wiener entropy ratio; white noise ≈ 1, structured signal ≪ 1

## Repository Structure

```
srpf-missile-defence/
├── paper/
│   ├── XJR_SRPF_Missile_Defence.pdf    # Full research paper (19 pages)
│   └── XJR_SRPF_Missile_Defence.tex    # LaTeX source
├── src/
│   └── srpf_monte_carlo.py             # Monte Carlo validation suite
├── figures/
│   └── SRPF_Monte_Carlo_Results.png    # 8-panel results figure
├── LICENSE
└── README.md
```

## Quick Start

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Run the full Monte Carlo validation

```bash
python src/srpf_monte_carlo.py
```

This runs 4 experiments (~2 minutes on a modern CPU):

- **Experiment 1** — Spectral Separation Theorem validation (500 trials × 3 classes)
- **Experiment 2** — SRPF score distributions & ROC curves (2,000 trials)
- **Experiment 3** — Computational gain factor G(τ) sweep (1,500 trials)
- **Experiment 4** — Saturation attack scenario: M=500, 50 threats, C=100 (500 replications)

Outputs a summary table and an 8-panel figure:

```
=================================================================
  SUMMARY
=================================================================
  PSD slopes:         -0.06 / -1.29 / -0.28
  AUC SRPF/Energy:    0.969 / 0.625
  Gain G(τ*):         ×2.75 (P_D=90%, filt=67%)
  Saturation missed:  30.9 vs 40.1
=================================================================
```

## Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Δt | 0.05 s | Radar sampling interval |
| β = m/(C_D·A) | 5,556 kg/m² | Ballistic coefficient |
| R_std | 15 m | Measurement noise (1σ) |
| Q_std | 0.3 m/s | Process noise (1σ, per axis) |
| N | 128 | Measurement window length |
| v₀ | [200, 400] m/s | Launch velocity range |
| θ | [35°, 70°] | Launch elevation range |
| σ_lift (decoys) | 50 m/s² | Aerodynamic lift perturbation |
| σ_wander (clutter) | 120 m/√s | Random walk intensity |

## Relation to Spectral Number Theory

SRPF is a direct transposition of the SSP (Spectral Sieve Pre-filtering) algorithm from integer factorisation:

| | SSP (Cryptanalysis) | SRPF (Missile Defence) |
|---|---|---|
| Structured objects | B-smooth integers | Ballistic trajectories |
| Unstructured objects | Non-smooth integers | Clutter / decoys |
| Spectral signature | \|F̂(k)\| = O(k^{-1/π(B)}) | E[\|δ̂(j)\|²] = O(j⁻¹) |
| Reference frequencies | Riemann zeta zeros {γⱼ} | Ballistic eigenfrequencies {ωⱼ} |
| Pre-filter gain | G(τ) = g(τ)/f(τ) | G(τ) = g(τ)/f(τ) |

For the foundational spectral number theory, see:
- [From Undecidability to Cryptographic Efficiency](https://doi.org/10.5281/zenodo.19002607) — SSP algorithm & Jinx's Theorem
- [The Jinx's Theorem — Source Code](https://github.com/carylun/jinx-theorem) — spectral fingerprinting implementation

## Citation

```bibtex
@article{Regent2026SRPF,
  author  = {R{\'e}gent, Xavier J.},
  title   = {Spectral Resonance Pre-Filtering for Ballistic Target
             Classification in Missile Defence Systems},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/carylun/srpf-missile-defence}
}
```

## License

MIT — see [LICENSE](./LICENSE).

## Author

**Xavier J. Régent** — Independent Researcher  
📧 xr@kerzu.org  
🔗 [ORCID: 0009-0005-9510-5335](https://orcid.org/0009-0005-9510-5335)
