# THESIS_REQUIREMENTS.md

## 1. Thesis background

The thesis studies a dynamic flexible job shop scheduling problem under multi-concurrent disturbances.

The key dynamic disturbance scenario is:

random job arrival + urgent order insertion

The purpose of the current code stage is to generate mid-term inspection evidence, not final large-scale experimental results.

## 2. Required dynamic job attributes

Each job must include:

- job_id
- n_i: number of operations
- RD_i: actual arrival time / release time
- DD_i: due date
- c_i: arrival type code
- w_i: priority weight
- completed flag
- current operation index

Arrival type code:

- c_i = 0: initial/static job
- c_i = 1: random arrival job
- c_i = 2: urgent insertion job

Priority weight:

- w_i = 1: stock order
- w_i = 2: regular production or random arrival order
- w_i = 4: urgent insertion order

## 3. Objective

Use the thesis objective:

F_real = sum_i w_i * abs(C_i - DD_i)

Where:

- C_i is the final completion time of job J_i
- DD_i is the due date of job J_i
- w_i is the priority weight

Urgent orders should not be forced to finish before DD_i as a hard constraint.

## 4. Ternary heterogeneous graph

The graph must contain three node types:

1. Job node J
2. Operation node O
3. Machine node M

The graph must contain two edge types:

1. J-O edge
   - connects each unfinished job to its current active operation
   - each unfinished job has exactly one active J-O edge
   - when an operation is scheduled, the J-O edge moves to the next operation
   - if all operations are completed, mark the job completed

2. O-M edge
   - connects ready operations to candidate machines
   - each O-M edge corresponds to one legal action
   - edge feature is processing time p_ijk

## 5. Node features

Job feature:

[n_i, l_i, ST_i, ERPT_i, RD_i, DD_i, c_i, w_i]

Operation feature:

[|K_ij|, EST_ij, EPT_ij]

Machine feature:

[AT_k, UR_k, |N_k|, APT_k]

Do not duplicate job-level features such as DD_i, c_i, and w_i into every operation node.

## 6. Dynamic update rules

When a random arrival job or urgent insertion job enters the system:

1. Add a Job node.
2. Add all Operation nodes of this job.
3. Add J_i -> O_i1 edge.
4. Add O_i1 -> M_k edges for all candidate machines M_k in K_i1.
5. Do not directly change machine AT or machine utilization.

When operation O_ij is scheduled:

1. Remove O_ij related O-M candidate edges from the current legal action graph.
2. If j < n_i:
   - move J_i -> O_ij to J_i -> O_i,j+1
   - add O_i,j+1 -> candidate machine edges
3. If j == n_i:
   - mark job completed
   - remove it from legal action candidates
   - preserve its historical features for objective and metrics

Machine AT and utilization update only after an operation is assigned and its completion time is calculated.

## 7. Action space

Action is:

a_t = (O_ij, M_k)

Legal action set:

A_t = {(O_ij, M_k) | O_ij is ready, M_k in K_ij}

The demo must output legal action count.

## 8. Mid-term demo requirement

Please create or improve the following files:

1. run_small_demo.py
2. results/graph_debug.txt
3. results/demo_metrics.csv
4. results/training_curve.png
5. README_Thesis.md

The demo must run by:

python run_small_demo.py

The demo scenario must be a small-scale concurrent disturbance case that includes:

- initial jobs
- random arrival jobs
- urgent insertion jobs

The console output must include:

1. initial job count
2. random arrival job count
3. urgent insertion job count
4. Job nodes count
5. Operation nodes count
6. Machine nodes count
7. J-O edges count
8. O-M edges count
9. legal actions count
10. F_real
11. makespan
12. urgent_tardiness
13. urgent_on_time_rate
14. machine_utilization
15. decision_time

The file results/graph_debug.txt must record graph changes before and after concurrent disturbance:

- Job nodes
- Operation nodes
- Machine nodes
- J-O edges
- O-M edges
- legal actions

The file results/demo_metrics.csv must save:

- scenario
- F_real
- makespan
- urgent_tardiness
- urgent_on_time_rate
- machine_utilization
- decision_time

The file results/training_curve.png must save a small demo curve using F_real or reward. It does not need to prove final convergence; it only needs to prove the result visualization pipeline works.

The README_Thesis.md file must explain:

- what this demo proves
- how to run it
- what output files are generated
- what is simplified
- what remains for future HGAT-PPO completion

## 9. Important constraints

- Do not break original PPO_model.py.
- Do not remove original results/*.pt model files.
- The demo can use simplified scheduling logic if HGAT-PPO is not fully implemented.
- The demo must be reproducible.
- Use fixed random seed.
- Use only small data so the script runs quickly.
- Prefer standard Python libraries plus existing project dependencies.
- If matplotlib is available, save training_curve.png.
- If any output folder is missing, create it automatically.
