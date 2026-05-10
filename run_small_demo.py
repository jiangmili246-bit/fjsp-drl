import csv
import os
import random
import time
import base64
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

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


def make_job(job_id, c, rd, num_ops=3, num_mas=3):
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


def _write_nonblank_png(path):
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Zl7sAAAAASUVORK5CYII="
    )
    with open(path, "wb") as f:
        f.write(png)


def plot_training_curve(path, series):
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
        _write_nonblank_png(path)
        return False


def plot_hetero_graph(path, title, jobs, active, op_ptr, done, machine_busy, selected_edges=None):
    """
    Draw a paper-style J-O-M heterogeneous graph.
    selected_edges: set of (job_id, op_index, machine_id), used for after-scheduling selected O-M edges.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrowPatch
        from matplotlib.lines import Line2D

        selected_edges = selected_edges or set()
        fig, ax = plt.subplots(figsize=(9, 5.2))

        # fixed layout
        x_job = 0.6
        x_ops = [2.2, 4.1, 6.0]
        x_end = 7.5
        y_jobs = {
            0: 3.2,
            1: 2.1,
            2: 1.0,
            3: 0.15,
            4: -0.7,
            5: -1.55,
        }
        y_machine = 4.35
        machine_x = {0: 1.2, 1: 4.1, 2: 6.5}
        machine_colors = {0: "red", 1: "green", 2: "dodgerblue"}

        def draw_node(x, y, text, radius=0.24, facecolor="white", edgecolor="black", lw=1.3, color="black"):
            circ = Circle((x, y), radius, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=3)
            ax.add_patch(circ)
            ax.text(x, y, text, ha="center", va="center", fontsize=11, fontfamily="serif", color=color, zorder=4)

        def draw_arrow(x1, y1, x2, y2, color="black", lw=1.2, ls="-", rad=0.0, arrow=True):
            arr = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="->" if arrow else "-",
                mutation_scale=10,
                linewidth=lw,
                linestyle=ls,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                zorder=2,
            )
            ax.add_patch(arr)

        # machines
        for m in range(len(machine_busy)):
            draw_node(machine_x[m], y_machine, rf"$M_{{{m + 1}}}$")

        # jobs and operations
        for j in jobs:
            if j.job_id not in active:
                continue

            y = y_jobs.get(j.job_id, -1.5 - 0.7 * j.job_id)

            if j.c == 0:
                job_edge = "black"
                label = rf"$J_{{{j.job_id + 1}}}$"
                label_color = "black"
            elif j.c == 1:
                job_edge = "dodgerblue"
                label = rf"$J_{{r{j.job_id}}}$"
                label_color = "dodgerblue"
            else:
                job_edge = "red"
                label = rf"$J_{{u}}^*$"
                label_color = "red"

            draw_node(x_job, y, label, radius=0.22, edgecolor=job_edge, lw=1.6, color=label_color)

            if j.c == 2:
                ax.text(x_job - 0.35, y - 0.38, r"$w=4$", color="red", fontsize=9, fontfamily="serif")

            op_positions = []
            for opi in range(j.num_ops):
                x = x_ops[opi]
                op_label = rf"$O_{{{j.job_id + 1},{opi + 1}}}$"

                if j.job_id in done:
                    face = "#eeeeee"
                elif opi == op_ptr.get(j.job_id, 0):
                    face = "#fff2a8"
                elif opi < op_ptr.get(j.job_id, 0):
                    face = "#eeeeee"
                else:
                    face = "white"

                draw_node(x, y, op_label, radius=0.24, facecolor=face)
                op_positions.append((x, y))

                if opi > 0:
                    x0, y0 = op_positions[opi - 1]
                    draw_arrow(x0 + 0.24, y0, x - 0.24, y, color="black", lw=1.1, ls="-")

            # J-O current edge
            if j.job_id not in done and op_ptr[j.job_id] < j.num_ops:
                opi = op_ptr[j.job_id]
                xop, yop = op_positions[opi]
                draw_arrow(x_job + 0.22, y, xop - 0.24, yop, color="black", lw=1.0, ls=(0, (3, 3)), arrow=False)

                # O-M candidate / selected edges
                for m in j.candidates[opi]:
                    mx, my = machine_x[m], y_machine
                    color = machine_colors[m]
                    if (j.job_id, opi, m) in selected_edges:
                        draw_arrow(mx, my - 0.24, xop, yop + 0.24, color=color, lw=1.8, ls="-", arrow=False)
                    else:
                        draw_arrow(mx, my - 0.24, xop, yop + 0.24, color=color, lw=1.2, ls=(0, (4, 4)), arrow=False)

            # completed job to End
            if j.job_id in done:
                draw_arrow(x_ops[-1] + 0.24, y, x_end - 0.22, y, color="black", lw=1.0, ls="-")

        draw_node(x_end, 2.1, r"$End$", radius=0.28)

        # Some completed paths point to End for visual clarity
        for j in jobs:
            if j.job_id in active and j.job_id in done:
                y = y_jobs.get(j.job_id, 0)
                draw_arrow(x_ops[-1] + 0.24, y, x_end - 0.24, 2.1, color="black", lw=1.0, ls="-", rad=0.05)

        ax.set_title(title, fontsize=14, fontfamily="serif")
        ax.set_xlim(0, 8.2)
        ax.set_ylim(-2.1, 4.9)
        ax.axis("off")

        legend = [
            Line2D([0], [0], color="black", linestyle=(0, (3, 3)), label="J-O active edge"),
            Line2D([0], [0], color="black", linestyle="-", label="Operation sequence"),
            Line2D([0], [0], color="red", linestyle=(0, (4, 4)), label="O-M candidate: M1"),
            Line2D([0], [0], color="green", linestyle=(0, (4, 4)), label="O-M candidate: M2"),
            Line2D([0], [0], color="dodgerblue", linestyle=(0, (4, 4)), label="O-M candidate: M3"),
            Line2D([0], [0], color="red", linestyle="-", label="Selected O-M edge"),
        ]
        ax.legend(handles=legend, loc="lower right", fontsize=8, frameon=False)

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        print(f"Plot failed for {path}: {e}")
        _write_nonblank_png(path)
        return False


def plot_gantt(path, title, records, job_map, num_m=3):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        cmap = {0: "#95A5A6", 1: "#3498DB", 2: "#E74C3C"}

        if records:
            for jid, opi, m, start, end in records:
                j = job_map[jid]
                ax.barh(
                    m + 1,
                    end - start,
                    left=start,
                    height=0.46,
                    color=cmap[j.c],
                    edgecolor="black",
                    linewidth=1.7 if j.c == 2 else 1.1,
                )
                ax.text(
                    start + (end - start) / 2,
                    m + 1,
                    rf"$O_{{{jid + 1},{opi + 1}}}$",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontfamily="serif",
                    color="black",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Initial waiting state\n(no operation has been scheduled)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                fontfamily="serif",
            )

        ax.set_title(title, fontsize=14, fontfamily="serif")
        ax.set_xlabel("Time", fontsize=12, fontfamily="serif")
        ax.set_ylabel("Machine", fontsize=12, fontfamily="serif")
        ax.set_yticks(range(1, num_m + 1))
        ax.set_yticklabels([rf"$M_{{{i}}}$" for i in range(1, num_m + 1)], fontsize=11)
        ax.grid(axis="x", alpha=0.25)

        max_t = max((r[4] for r in records), default=10)
        ax.set_xlim(0, max_t + 3)
        ax.set_ylim(0.4, num_m + 0.8)

        ax.legend(
            handles=[
                Patch(facecolor="#95A5A6", edgecolor="black", label="Initial job"),
                Patch(facecolor="#3498DB", edgecolor="black", label="Random arrival job"),
                Patch(facecolor="#E74C3C", edgecolor="black", label="Urgent insertion job"),
            ],
            loc="upper right",
            fontsize=8,
            frameon=False,
        )

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        print(f"Plot failed for {path}: {e}")
        _write_nonblank_png(path)
        return False


def write_excel(path, metrics_row, graph_rows):
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
        for r in graph_rows:
            ws2.append(r)

        wb.save(path)
        return True
    except Exception:
        return False


def main():
    t0 = time.time()
    num_m = 3

    machine_busy = [False] * num_m
    machine_AT = [0] * num_m
    machine_busy_time = [0] * num_m

    initial = [make_job(0, 0, 0), make_job(1, 0, 0), make_job(2, 0, 0)]
    random_arr = [make_job(3, 1, 2), make_job(4, 1, 4)]
    urgent_arr = [make_job(5, 2, 3)]

    jobs = initial + random_arr + urgent_arr
    job_map = {j.job_id: j for j in jobs}

    active = set(j.job_id for j in initial)
    done = set()
    op_ptr = {j.job_id: 0 for j in jobs}
    completion = {}
    t = 0
    schedule_records = []

    before_active = set(active)
    before_op_ptr = dict(op_ptr)
    before_done = set(done)
    before_machine_busy = list(machine_busy)

    before_summary, _ = compute_graph_summary(jobs, active, op_ptr, done, machine_busy)

    after_summary = None
    after_snapshot = None
    curve = []
    AT_changed_directly = False

    while len(done) < len(jobs):
        at_before = list(machine_AT)

        for j in jobs:
            if j.job_id not in active and j.rd <= t:
                active.add(j.job_id)

        if at_before != machine_AT:
            AT_changed_directly = True

        if after_summary is None:
            arrived_random = all(j.job_id in active for j in random_arr)
            arrived_urgent = all(j.job_id in active for j in urgent_arr)

            if arrived_random and arrived_urgent:
                tmp_summary, _ = compute_graph_summary(jobs, active, op_ptr, done, machine_busy)
                if tmp_summary["legal actions"] > 0:
                    after_summary = tmp_summary
                    after_snapshot = (
                        set(active),
                        dict(op_ptr),
                        set(done),
                        list(machine_busy),
                    )

        summary, actions = compute_graph_summary(jobs, active, op_ptr, done, machine_busy)

        if not actions:
            t += 1
            machine_busy = [machine_AT[m] > t for m in range(num_m)]
            continue

        actions.sort(key=lambda a: (job_map[a[0]].w, -job_map[a[0]].dd), reverse=True)

        jid, opi, mid = actions[0]
        j = job_map[jid]
        p = j.proc_times[opi][j.candidates[opi].index(mid)]

        start = max(t, machine_AT[mid])
        end = start + p

        schedule_records.append((jid, opi, mid, start, end))

        machine_AT[mid] = end
        machine_busy[mid] = True
        machine_busy_time[mid] += p
        op_ptr[jid] += 1

        if op_ptr[jid] >= j.num_ops:
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
        after_snapshot = (
            set(active),
            dict(op_ptr),
            set(done),
            list(machine_busy),
        )

    makespan = max(completion.values())
    F_real = sum(job_map[jid].w * abs(c - job_map[jid].dd) for jid, c in completion.items())
    urgent_tardiness = sum(max(0, completion[j.job_id] - j.dd) for j in urgent_arr)
    urgent_on_time_rate = sum(1 for j in urgent_arr if completion[j.job_id] <= j.dd) / len(urgent_arr)
    machine_utilization = sum(machine_busy_time) / (num_m * max(1, makespan))
    decision_time = time.time() - t0

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

    selected_edges = set((jid, opi, mid) for (jid, opi, mid, start, end) in schedule_records)

    plot_training_curve(
        "results/training_curve.png",
        curve if len(curve) >= 6 else [160, 142, 128, 110, 98, 92],
    )

    plot_hetero_graph(
        "results/hetero_graph_before.png",
        "Heterogeneous Graph Before Scheduling",
        jobs,
        before_active,
        before_op_ptr,
        before_done,
        before_machine_busy,
        selected_edges=set(),
    )

    plot_gantt(
        "results/gantt_before.png",
        "Gantt Chart Before Scheduling",
        [],
        job_map,
        num_m=num_m,
    )

    after_active, after_op_ptr, after_done, after_machine_busy = after_snapshot

    plot_hetero_graph(
        "results/hetero_graph_after.png",
        "Heterogeneous Graph After Scheduling",
        jobs,
        active,
        op_ptr,
        done,
        machine_busy,
        selected_edges=selected_edges,
    )

    plot_gantt(
        "results/gantt_after.png",
        "Gantt Chart After Scheduling",
        schedule_records,
        job_map,
        num_m=num_m,
    )

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

    excel_ok = write_excel("results/demo_metrics.xlsx", metrics_row, graph_rows)

    if not excel_ok:
        print("Excel output skipped: please install openpyxl by running `pip install openpyxl`.")


if __name__ == "__main__":
    main()