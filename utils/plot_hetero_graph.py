from dataclasses import dataclass

@dataclass
class HeteroSnapshot:
    jobs: list
    active_jobs: set
    done_jobs: set
    op_ptr: dict
    num_m: int
    selected_edges: set  # (job_id, op_i, m)
    phase: str           # initial/running/finished


def build_hetero_graph_snapshot(jobs, active_jobs, done_jobs, op_ptr, num_m, selected_edges=None, phase='running'):
    return HeteroSnapshot(
        jobs=jobs,
        active_jobs=set(active_jobs),
        done_jobs=set(done_jobs),
        op_ptr=dict(op_ptr),
        num_m=num_m,
        selected_edges=set(selected_edges or set()),
        phase=phase,
    )


def _jlabel(j):
    if j.c == 0:
        return rf"$J_{j.job_id+1}$"
    if j.c == 1:
        return rf"$J_{j.job_id+1}^r$"
    return rf"$J_{j.job_id+1}^{{u*}}$"


def draw_hetero_graph(snapshot: HeteroSnapshot, save_path: str):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    jobs = [j for j in snapshot.jobs if j.job_id in snapshot.active_jobs] if snapshot.phase != 'finished' else list(snapshot.jobs)
    jobs = sorted(jobs, key=lambda x: x.job_id)
    n_jobs = len(jobs)
    max_ops = max(j.num_ops for j in jobs)

    fig_w = max(14, 3 + max_ops * 2)
    fig_h = max(6, 2 + n_jobs * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # fixed layout
    x_job = 0.0
    x_ops = [2.2 + i * 1.8 for i in range(max_ops)]
    x_end = x_ops[-1] + 2.2
    x_m = [2.4 + i * 2.2 for i in range(snapshot.num_m)]
    y_m = n_jobs + 1.5

    y_of = {}
    for idx, j in enumerate(jobs):
        y_of[j.job_id] = n_jobs - idx

    # draw machines
    m_colors = ['red', 'green', 'dodgerblue', 'purple', 'orange', 'brown']
    for m in range(snapshot.num_m):
        ax.scatter(x_m[m], y_m, s=320, marker='^', facecolor='white', edgecolor='black', linewidth=1.2, zorder=3)
        ax.text(x_m[m], y_m + 0.25, rf"$M_{m+1}$", ha='center', fontsize=12)

    # draw end
    ax.scatter(x_end, (n_jobs + 1)/2, s=300, marker='D', facecolor='white', edgecolor='black', linewidth=1.2, zorder=3)
    ax.text(x_end, (n_jobs + 1)/2 + 0.25, 'End', ha='center', fontsize=12)

    # draw jobs & operations
    for j in jobs:
        y = y_of[j.job_id]
        jc = '#DDDDDD' if j.c == 0 else ('#BFDFFF' if j.c == 1 else '#FFCCCC')
        ax.scatter(x_job, y, s=300, marker='s', facecolor=jc, edgecolor='black', linewidth=1.2, zorder=3)
        ax.text(x_job, y + 0.25, _jlabel(j), ha='center', fontsize=11)
        if j.c == 2:
            ax.text(x_job + 0.35, y - 0.25, r'$w=4$', color='red', fontsize=10)

        # operations + sequence edges
        for oi in range(j.num_ops):
            x = x_ops[oi]
            active = (snapshot.phase != 'finished' and j.job_id not in snapshot.done_jobs and snapshot.op_ptr.get(j.job_id, 0) == oi)
            face = '#F9E79F' if active else ('#E5E7E9' if snapshot.phase == 'finished' else 'white')
            ax.scatter(x, y, s=250, marker='o', facecolor=face, edgecolor='black', linewidth=1.1, zorder=3)
            ax.text(x, y + 0.22, rf"$O_{{{j.job_id+1},{oi+1}}}$", ha='center', fontsize=9)
            if oi < j.num_ops - 1:
                ax.annotate('', xy=(x_ops[oi+1]-0.22, y), xytext=(x+0.22, y),
                            arrowprops=dict(arrowstyle='->', lw=1.1, color='black'))

        # completion edge style
        if snapshot.phase == 'finished':
            # from last op to End
            ax.plot([x_ops[j.num_ops-1]+0.2, x_end-0.25], [y, (n_jobs + 1)/2], '-', color='black', lw=1.0, alpha=0.7)
            # J->End dashed
            ax.plot([x_job+0.2, x_end-0.3], [y, (n_jobs + 1)/2], '--', color='black', lw=0.9, alpha=0.7)
        else:
            # J-O active pointer
            if j.job_id not in snapshot.done_jobs:
                p = min(snapshot.op_ptr.get(j.job_id, 0), j.num_ops-1)
                ax.plot([x_job+0.2, x_ops[p]-0.2], [y, y], '--', color='black', lw=1.1)

        # O-M edges by phase
        if snapshot.phase in ('initial', 'running') and j.job_id not in snapshot.done_jobs:
            p = min(snapshot.op_ptr.get(j.job_id, 0), j.num_ops-1)
            for m in j.candidates[p]:
                ax.plot([x_m[m], x_ops[p]], [y_m-0.1, y+0.05], '--', color=m_colors[m % len(m_colors)], lw=1.0)
        if snapshot.phase == 'finished':
            for (jid, oi, m) in snapshot.selected_edges:
                if jid == j.job_id and oi < j.num_ops:
                    ax.plot([x_m[m], x_ops[oi]], [y_m-0.1, y+0.05], '-', color=m_colors[m % len(m_colors)], lw=1.6, alpha=0.75)

    if snapshot.phase == 'initial':
        ttl = '(a) 起始状态 / Initial state'
    elif snapshot.phase == 'running':
        ttl = '(b) 调度进行中 / During scheduling'
    else:
        ttl = '(c) 调度结束 / Scheduling finished'
    ax.set_title(ttl, fontsize=13)

    legend = [
        Line2D([0],[0], linestyle='--', color='black', label='J-O edge'),
        Line2D([0],[0], linestyle='-', color='black', label='operation sequence edge'),
        Line2D([0],[0], linestyle='--', color='red', label='O-M candidate edge (M1 red / M2 green / M3 blue)'),
        Line2D([0],[0], linestyle='-', color='red', label='selected O-M edge')
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=9)

    ax.set_xlim(-0.7, x_end + 0.8)
    ax.set_ylim(0.3, y_m + 0.8)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
