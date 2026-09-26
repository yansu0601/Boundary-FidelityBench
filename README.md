# Boundary-FidelityBench

Anonymous core-code supplement for **Boundary-FidelityBench: Probing First-Crossing Fidelity in Action-Conditioned World Models**.

This package contains selected numerical implementations of FCRG for code review.
It is not a complete experiment or reproduction package. The complete codebase is
planned for public release after acceptance.

## Reading guide

| Paper component | File | Main entry point |
| --- | --- | --- |
| Local Continuation Residual (LCR), Section 4.1 | `evidence.py` | `radius_innovation` |
| Scale-Aware Evidence (SAE), Section 4.1 | `evidence.py` | `fit_interval_envelope`, `residual_features` |
| Residual Evidence Fusion (REF), Section 4.2 | `model.py` | `FullSequenceRayHead` |
| Factorized Boundary Readout (FBR), Section 4.2 | `posterior.py` | `factorized_posterior` |

