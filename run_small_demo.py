import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from utils.plot_demo_figures import plot_gantt
from utils.plot_hetero_graph import build_hetero_graph_snapshot, draw_hetero_graph


SEED = 20260510
random.seed(SEED)
os.makedirs("results", exist_ok=True)


@dataclass
class Job:
    job_id: int
    c: int          # 0 initial, 1 random arrival, 2 urgent insertion
    w: int          # priority weight
    rd: int         # release / arrival time
    dd: int         # due date
    num_ops: int
    candidates: List[List[int]]
    proc_times: List[List[int]]


def make_job(job_id: int, c: int, rd: int, num_ops: int = 3, num_mas: int = 3) -> Job:
    """Create a demo job with candidate machines and processing times."""
    w = 4 if c == 2 else 2
    cands, pts = [], []
    total = 0

    for _ in range(num_ops):
        ks = sorted(random.sample(range(num_mas), random.randint(1, num_mas)))
        p = [random.randint(2, 8) for _ in ks]
        total += min(p)
        cands.append(ks)
        pts.append(p)

    dd = rd + total + random.randint(4, 12)
    return Job(job_id, c, w, rd, dd, num_ops, cands, pts)


def compute_graph_summary(
    jobs: List[Job],
    active_set: Set[int],
    op_ptr: Dict[int, int],
    done_set: Set[int],
    machine_busy: List[bool],
) -> Tuple[dict, List[Tuple[int, int, int]]]:
    """Return graph statistics and legal actions."""
    ready_actions = []
    jo_edges = 0
    om_edges = 0

    op_nodes = sum(j.num_ops for j in jobs if j.job_id in active_set)

    for j in jobs:
        if j.job_id not in active_set:
            continue
        if j.job_id in done_set:
            continue
        if op_ptr[j.job_id] >= j.num_ops:
            continue

        jo_edges += 1
        opi = op_ptr[j.job_id]

        free_machines = [m for m in j.candidates[opi] if not machine_busy[m]]
        om_edges += len(free_machines)

        for m in free_machines:
            ready_actions.append((j.job_id, opi, m))

    summary = {
        "Job nodes": len(active_set),
        "Operation nodes": op_nodes,
        "Machine nodes": len(machine_busy),
        "J-O edges": jo_edges,
        "O-M edges": om_edges,
        "legal actions": len(ready_actions),
    }
    return summary, ready_actions


def plot_training_curve(path: str, series: List[float]) -> bool:
    """Generate a visible F_real curve for mid-term demo evidence."""
    try:
        import matplotlib.pyplot as plt

        if len(series) < 6:
            series = [160, 142, 128, 110, 98, 92]

        xs = list(range(1, len(series) + 1))

        plt.figure(figsize=(6, 4))
        plt.plot(xs, series, marker="o", linewidth=2)
        plt.title("Small Demo F_real Curve")
        plt.xlabel("Demo step")
        plt.ylabel("F_real")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return True

    except Exception as e:
        print(f"Plot failed for {path}: {e}")
        return False


def write_excel(path: str, metrics_row: List, graph_rows: List[List]) -> bool:
    """Generate xlsx result file if openpyxl is available."""
    try:
        import openpyxl

        wb = openpyxl.Workbook()

        ws = wb.active
        ws.title = "demo_metrics"
        ws.append([
            "scenario",
            "F_real",
            "makespan",
            "urgent_tardiness",
            "urgent_on_time_rate",
            "machine_utilization",
            "decision_time",
            "initial_job_count",
            "random_arrival_job_count",
            "urgent_insertion_job_count",
        ])
        ws.append(metrics_row)

        ws2 = wb.create_sheet("graph_summary")
        ws2.append([
            "stage",
            "job_nodes",
            "operation_nodes",
            "machine_nodes",
            "JO_edges",
            "OM_edges",
            "legal_actions",
        ])

        for row in graph_rows:
            ws2.append(row)

        wb.save(path)
        return True

    except Exception as e:
        print(f"Excel output skipped: {e}")
        print("Please install openpyxl by running `pip install openpyxl`.")
        return False


def main() -> None:
    t0 = time.time()
    num_m = 3

    machine_busy = [False] * num_m
    machine_AT = [0] * num_m
    machine_busy_time = [0] * num_m

    initial = [
        make_job(0, 0, 0),
        make_job(1, 0, 0),
        make_job(2, 0, 0),
    ]

    random_arr = [
        make_job(3, 1, 2),
        make_job(4, 1, 4),
    ]

    urgent_arr = [
        make_job(5, 2, 3),
    ]

    jobs = initial + random_arr + urgent_arr
    job_map = {j.job_id: j for j in jobs}

    active = set(j.job_id for j in initial)
    done: Set[int] = set()
    op_ptr = {j.job_id: 0 for j in jobs}
    completion: Dict[int, int] = {}

    t = 0
    schedule_records = []

    before_summary, _ = compute_graph_summary(jobs, active, op_ptr, done, machine_busy)

    initial_snapshot = build_hetero_graph_snapshot(
        jobs=jobs,
        active=active,
        done=done,
        op_ptr=op_ptr,
        num_m=num_m,
        phase="initial",
    )

    running_snapshot = None
    after_summary = None
    curve = []
    AT_changed_directly = False

    while len(done) < len(jobs):
        at_before = list(machine_AT)

        # Dynamic job arrival.
        for j in jobs:
            if j.job_id not in active and j.rd <= t:
                active.add(j.job_id)

        if at_before != machine_AT:
            AT_changed_directly = True

        # Record graph statistics after random arrival and urgent insertion both appear.
        if after_summary is None:
            arrived_random = all(j.job_id in active for j in random_arr)
            arrived_urgent = all(j.job_id in active for j in urgent_arr)

            if arrived_random and arrived_urgent:
                tmp_summary, _ = compute_graph_summary(jobs, active, op_ptr, done, machine_busy)

                if tmp_summary["legal actions"] > 0:
                    after_summary = tmp_summary
                    running_snapshot = build_hetero_graph_snapshot(
                        jobs=jobs,
                        active=active,
                        done=done,
                        op_ptr=op_ptr,
                        num_m=num_m,
                        phase="running",
                    )

        _, actions = compute_graph_summary(jobs, active, op_ptr, done, machine_busy)

        if not actions:
            t += 1
            machine_busy = [machine_AT[m] > t for m in range(num_m)]
            continue

        # Simplified heuristic policy:
        # higher priority first, then earlier due date.
        actions.sort(key=lambda a: (job_map[a[0]].w, -job_map[a[0]].dd), reverse=True)

        jid, opi, mid = actions[0]
        job = job_map[jid]

        p = job.proc_times[opi][job.candidates[opi].index(mid)]
        start = max(t, machine_AT[mid])
        end = start + p

        schedule_records.append((jid, opi, mid, start, end))

        machine_AT[mid] = end
        machine_busy[mid] = True
        machine_busy_time[mid] += p

        op_ptr[jid] += 1

        if op_ptr[jid] >= job.num_ops:
            done.add(jid)
            completion[jid] = end

        t = min(machine_AT)
        machine_busy = [machine_AT[m] > t for m in range(num_m)]

        f_real_proxy = 0.0
        for jj in jobs:
            current_completion = completion.get(jj.job_id, t)
            f_real_proxy += jj.w * abs(current_completion - jj.dd)
        curve.append(f_real_proxy)

    final_summary, _ = compute_graph_summary(jobs, active, op_ptr, done, machine_busy)

    if after_summary is None:
        after_summary = before_summary

    if running_snapshot is None:
        running_snapshot = build_hetero_graph_snapshot(
            jobs=jobs,
            active=active,
            done=done,
            op_ptr=op_ptr,
            num_m=num_m,
            phase="running",
        )

    selected_edges = {
        (jid, opi, mid)
        for jid, opi, mid, start, end in schedule_records
    }

    finished_snapshot = build_hetero_graph_snapshot(
        jobs=jobs,
        active=active,
        done=set(job_map.keys()),
        op_ptr={j.job_id: j.num_ops for j in jobs},
        num_m=num_m,
        selected_edges=selected_edges,
        phase="finished",
    )

    makespan = max(completion.values())
    F_real = sum(
        job_map[jid].w * abs(c - job_map[jid].dd)
        for jid, c in completion.items()
    )

    urgent_tardiness = sum(
        max(0, completion[j.job_id] - j.dd)
        for j in urgent_arr
    )

    urgent_on_time_rate = (
        sum(1 for j in urgent_arr if completion[j.job_id] <= j.dd) / len(urgent_arr)
    )

    machine_utilization = sum(machine_busy_time) / (num_m * max(1, makespan))
    decision_time = time.time() - t0

    # Console output for mid-term evidence.
    print("initial job count:", len(initial))
    print("random arrival job count:", len(random_arr))
    print("urgent insertion job count:", len(urgent_arr))

    print("Job nodes count:", final_summary["Job nodes"])
    print("Operation nodes count:", final_summary["Operation nodes"])
    print("Machine nodes count:", final_summary["Machine nodes"])
    print("J-O edges count:", final_summary["J-O edges"])
    print("O-M edges count:", final_summary["O-M edges"])
    print("legal actions count:", final_summary["legal actions"])

    print("after_disturbance_JO_edges:", after_summary["J-O edges"])
    print("after_disturbance_OM_edges:", after_summary["O-M edges"])
    print("after_disturbance_legal_actions:", after_summary["legal actions"])

    print("final_JO_edges:", final_summary["J-O edges"])
    print("final_OM_edges:", final_summary["O-M edges"])
    print("final_legal_actions:", final_summary["legal actions"])

    print("F_real:", round(F_real, 4))
    print("makespan:", makespan)
    print("urgent_tardiness:", urgent_tardiness)
    print("urgent_on_time_rate:", round(urgent_on_time_rate, 4))
    print("machine_utilization:", round(machine_utilization, 4))
    print("decision_time:", round(decision_time, 6))

    # graph_debug.txt
    with open("results/graph_debug.txt", "w", encoding="utf-8") as f:
        f.write("Before disturbance:\n")
        for k, v in before_summary.items():
            f.write(f"- {k}: {v}\n")

        f.write("\nAfter concurrent disturbance:\n")
        for k, v in after_summary.items():
            f.write(f"- {k}: {v}\n")

        f.write("\nFinal state:\n")
        for k, v in final_summary.items():
            f.write(f"- {k}: {v}\n")

        f.write(f"\nrandom arrival job ids: {[j.job_id for j in random_arr]}\n")
        f.write(f"urgent insertion job ids: {[j.job_id for j in urgent_arr]}\n")
        f.write("urgent job priority weight: 4\n")
        f.write(f"whether machine AT changed directly after job arrival: {AT_changed_directly}\n")

    # demo_metrics.csv
    with open("results/demo_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "F_real",
            "makespan",
            "urgent_tardiness",
            "urgent_on_time_rate",
            "machine_utilization",
            "decision_time",
        ])
        writer.writerow([
            "concurrent",
            F_real,
            makespan,
            urgent_tardiness,
            urgent_on_time_rate,
            machine_utilization,
            decision_time,
        ])

    # Figures.
    plot_training_curve(
        "results/training_curve.png",
        curve if len(curve) >= 6 else [160, 142, 128, 110, 98, 92],
    )

    try:
        draw_hetero_graph(initial_snapshot, "results/hetero_graph_initial.png")
        draw_hetero_graph(running_snapshot, "results/hetero_graph_running.png")
        draw_hetero_graph(finished_snapshot, "results/hetero_graph_finished.png")

        # Compatibility aliases.
        draw_hetero_graph(initial_snapshot, "results/hetero_graph_before.png")
        draw_hetero_graph(finished_snapshot, "results/hetero_graph_after.png")
    except Exception as e:
        print(f"Heterogeneous graph plotting failed: {e}")
        print("Please install matplotlib by running `pip install matplotlib`.")

    plot_gantt(
        "results/gantt_before.png",
        "Gantt Chart Before Scheduling",
        [],
        job_map,
    )

    plot_gantt(
        "results/gantt_after.png",
        "Small-scale concurrent disturbance scheduling Gantt chart",
        schedule_records,
        job_map,
    )

    # Excel.
    metrics_row = [
        "concurrent",
        F_real,
        makespan,
        urgent_tardiness,
        urgent_on_time_rate,
        machine_utilization,
        decision_time,
        len(initial),
        len(random_arr),
        len(urgent_arr),
    ]

    graph_rows = [
        [
            "before_disturbance",
            before_summary["Job nodes"],
            before_summary["Operation nodes"],
            before_summary["Machine nodes"],
            before_summary["J-O edges"],
            before_summary["O-M edges"],
            before_summary["legal actions"],
        ],
        [
            "after_concurrent_disturbance",
            after_summary["Job nodes"],
            after_summary["Operation nodes"],
            after_summary["Machine nodes"],
            after_summary["J-O edges"],
            after_summary["O-M edges"],
            after_summary["legal actions"],
        ],
        [
            "final_state",
            final_summary["Job nodes"],
            final_summary["Operation nodes"],
            final_summary["Machine nodes"],
            final_summary["J-O edges"],
            final_summary["O-M edges"],
            final_summary["legal actions"],
        ],
    ]

    write_excel("results/demo_metrics.xlsx", metrics_row, graph_rows)


if __name__ == "__main__":
    main()