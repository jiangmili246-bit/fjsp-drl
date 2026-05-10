# AGENTS.md

## Project instruction

This repository is being modified from an existing FJSP-DRL implementation into a thesis-oriented DFJSP-MD demo.

Before making code changes, always read:

- THESIS_REQUIREMENTS.md
- README.md
- config.json
- PPO_model.py
- env/
- graph/
- model/
- utils/

## Hard rules

1. Do not break the original PPO_model.py.
2. Do not delete original folders or model files.
3. If HGAT-PPO is not fully implemented, build a simplified runnable demo first.
4. The priority of this stage is reproducibility for mid-term thesis inspection, not final algorithm performance.
5. Any new demo must be runnable by:
   python run_small_demo.py
6. All output files must be saved under results/.
7. If assumptions are made, write them clearly in README_Thesis.md.

## Thesis target

The target thesis code should demonstrate:

- dynamic job arrival
- urgent order insertion
- concurrent random arrival + urgent insertion
- job-operation-machine ternary heterogeneous graph
- J-O edge update
- O-M candidate edge update
- legal action generation and action mask
- weighted earliness/tardiness objective

The objective is:

F_real = sum_i w_i * abs(C_i - DD_i)

Urgent orders must not be implemented as hard due-date constraints. Their priority is represented by w_i and job node features.

---
