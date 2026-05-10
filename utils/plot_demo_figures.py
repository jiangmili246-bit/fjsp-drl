import base64


def _fallback_png(path):
    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Zl7sAAAAASUVORK5CYII=')
    with open(path, 'wb') as f:
        f.write(png)


def plot_hetero_graph(path, title, jobs, active_set, op_ptr, done_set, selected_edges=None):
    selected_edges = selected_edges or set()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(11, 6))
        # layout
        job_x = 0.1
        op_x = 0.45
        mach_x = 0.82
        ys = {}
        active_jobs = [j for j in jobs if j.job_id in active_set]
        for idx, j in enumerate(active_jobs):
            ys[j.job_id] = 0.9 - idx * 0.12

        # machine nodes top row-like right side
        num_m = max(max((max(ops) if ops else 0) for ops in [c for j in jobs for c in j.candidates]), default=0) + 1
        my = {m: 0.2 + m * 0.2 for m in range(num_m)}

        # job nodes
        for j in active_jobs:
            color = '#AAB7B8' if j.c == 0 else ('#3498DB' if j.c == 1 else '#E74C3C')
            ax.scatter(job_x, ys[j.job_id], s=280, marker='s', color=color, edgecolors='black', zorder=3)
            ax.text(job_x, ys[j.job_id] + 0.035, f'J{j.job_id}', ha='center', fontsize=9)

        # operation nodes & edges
        for j in active_jobs:
            y = ys[j.job_id]
            for oi in range(j.num_ops):
                ox = op_x + oi * 0.08
                alpha = 1.0 if oi == op_ptr.get(j.job_id, 0) and j.job_id not in done_set else 0.5
                ax.scatter(ox, y, s=200, marker='o', color='#F7DC6F', edgecolors='black', alpha=alpha, zorder=3)
                ax.text(ox, y + 0.03, f'O{j.job_id},{oi+1}', ha='center', fontsize=7)
                if oi < j.num_ops - 1:
                    ax.annotate('', xy=(op_x + (oi+1) * 0.08 - 0.015, y), xytext=(ox + 0.015, y),
                                arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))
            # J-O dashed pointer
            if j.job_id not in done_set and op_ptr.get(j.job_id, 0) < j.num_ops:
                cur = op_ptr[j.job_id]
                ax.plot([job_x + 0.02, op_x + cur * 0.08 - 0.02], [y, y], linestyle='--', color='black', lw=1.0)

                # candidate O-M edges
                cands = j.candidates[cur]
                colors = ['#E74C3C', '#2ECC71', '#3498DB', '#9B59B6']
                for m in cands:
                    line_style = '-' if (j.job_id, cur, m) in selected_edges else '--'
                    ax.plot([op_x + cur * 0.08 + 0.01, mach_x - 0.02], [y, my[m]],
                            linestyle=line_style, color=colors[m % len(colors)], lw=1.2)

        for m, y in my.items():
            ax.scatter(mach_x, y, s=260, marker='^', color='#58D68D', edgecolors='black', zorder=3)
            ax.text(mach_x, y + 0.03, f'M{m+1}', ha='center', fontsize=9)

        legend = [
            Line2D([0], [0], marker='s', color='w', label='Initial Job', markerfacecolor='#AAB7B8', markeredgecolor='black', markersize=9),
            Line2D([0], [0], marker='s', color='w', label='Random Arrival Job', markerfacecolor='#3498DB', markeredgecolor='black', markersize=9),
            Line2D([0], [0], marker='s', color='w', label='Urgent Insertion Job', markerfacecolor='#E74C3C', markeredgecolor='black', markersize=9),
            Line2D([0], [0], marker='o', color='w', label='Operation', markerfacecolor='#F7DC6F', markeredgecolor='black', markersize=9),
            Line2D([0], [0], marker='^', color='w', label='Machine', markerfacecolor='#58D68D', markeredgecolor='black', markersize=9),
            Line2D([0], [0], linestyle='-', color='black', label='Operation Sequence Edge'),
            Line2D([0], [0], linestyle='--', color='black', label='J-O Edge (active pointer)'),
            Line2D([0], [0], linestyle='--', color='#3498DB', label='O-M Candidate Edge'),
            Line2D([0], [0], linestyle='-', color='#3498DB', label='Selected O-M Edge'),
        ]
        ax.legend(handles=legend, loc='upper right', fontsize=8)
        ax.set_title(title)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception:
        _fallback_png(path)
        return False


def plot_gantt(path, title, schedule_records, job_map):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        fig, ax = plt.subplots(figsize=(11, 5))
        if not schedule_records:
            # initial waiting-state timeline
            for m in range(3):
                ax.barh(m, 1, left=0, color='white', edgecolor='black', linewidth=1.0)
            ax.text(0.5, 1.0, 'Initial waiting-state (no assigned operations yet)', fontsize=10)
        else:
            cmap = {0: '#95A5A6', 1: '#3498DB', 2: '#E74C3C'}
            for jid, opi, m, st, en in schedule_records:
                j = job_map[jid]
                lw = 2.2 if j.c == 2 else 1.0
                ax.barh(m, en - st, left=st, color=cmap[j.c], edgecolor='black', linewidth=lw)
                ax.text(st + (en - st)/2, m, f'O_{{{jid},{opi+1}}}', ha='center', va='center', fontsize=8, color='white')
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        handles = [
            Patch(facecolor='#95A5A6', edgecolor='black', label='Initial Job'),
            Patch(facecolor='#3498DB', edgecolor='black', label='Random Arrival Job'),
            Patch(facecolor='#E74C3C', edgecolor='black', label='Urgent Insertion Job'),
        ]
        ax.legend(handles=handles, loc='upper right')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception:
        _fallback_png(path)
        return False
