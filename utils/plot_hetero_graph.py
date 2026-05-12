from dataclasses import dataclass


def _fallback_png(path):
    import struct, zlib
    w, h = 480, 320
    row = bytes([255, 255, 255]) * w
    img = [bytearray(row) for _ in range(h)]
    raw = b''.join(b'\x00'+bytes(r) for r in img)
    comp = zlib.compress(raw, 9)
    def chunk(tag,data):
        return struct.pack('!I',len(data))+tag+data+struct.pack('!I',zlib.crc32(tag+data)&0xffffffff)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', w,h,8,2,0,0,0)) + chunk(b'IDAT', comp)+chunk(b'IEND', b'')
    open(path,'wb').write(png)


@dataclass
class HeteroSnapshot:
    jobs: list
    active_jobs: set
    done_jobs: set
    op_ptr: dict
    num_m: int
    selected_edges: set
    phase: str  # initial/disturbance/after


def build_hetero_graph_snapshot(jobs, active_jobs, done_jobs, op_ptr, num_m, selected_edges=None, phase='disturbance'):
    return HeteroSnapshot(
        jobs=jobs,
        active_jobs=set(active_jobs),
        done_jobs=set(done_jobs),
        op_ptr=dict(op_ptr),
        num_m=num_m,
        selected_edges=set(selected_edges or set()),
        phase=phase,
    )


def draw_hetero_graph(snapshot: HeteroSnapshot, save_path: str, state_name: str = ""):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _fallback_png(save_path)
        return False

    if snapshot.phase == 'after':
        jobs = sorted(snapshot.jobs, key=lambda x: x.job_id)
    else:
        jobs = sorted([j for j in snapshot.jobs if j.job_id in snapshot.active_jobs], key=lambda x: x.job_id)
    n_jobs = len(jobs)
    max_ops = max(j.num_ops for j in jobs)

    fig, ax = plt.subplots(figsize=(10, max(4.8, 1.0 + n_jobs * 0.8)))

    x_job = 0.5
    x_ops = [2.0 + i * 1.7 for i in range(max_ops)]
    x_end = x_ops[-1] + 2.0
    x_m = [2.0 + i * 1.8 for i in range(snapshot.num_m)]
    y_m = n_jobs + 1.2

    y_of = {j.job_id: n_jobs - idx for idx, j in enumerate(jobs)}

    m_colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']

    for m in range(snapshot.num_m):
        ax.scatter(x_m[m], y_m, s=300, marker='^', facecolor='white', edgecolor='black', linewidth=1.0)
        ax.text(x_m[m], y_m + 0.22, f"M{m+1}", ha='center', fontsize=10)

    y_end = (n_jobs + 1) / 2
    ax.scatter(x_end, y_end, s=260, marker='D', facecolor='white', edgecolor='black', linewidth=1.0)
    ax.text(x_end, y_end + 0.22, 'End', ha='center', fontsize=10)

    for j in jobs:
        y = y_of[j.job_id]
        edge = 'black' if j.c == 0 else ('blue' if j.c == 1 else 'red')
        ax.scatter(x_job, y, s=260, marker='s', facecolor='white', edgecolor=edge, linewidth=1.2)
        ax.text(x_job, y + 0.2, f"J{j.job_id+1}", ha='center', fontsize=9)

        for oi in range(j.num_ops):
            ox = x_ops[oi]
            ax.scatter(ox, y, s=220, marker='o', facecolor='white', edgecolor='black', linewidth=1.0)
            ax.text(ox, y + 0.2, f"O{j.job_id+1},{oi+1}", ha='center', fontsize=8)
            if oi < j.num_ops - 1:
                ax.annotate('', xy=(x_ops[oi + 1] - 0.2, y), xytext=(ox + 0.2, y),
                            arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))
        ax.annotate('', xy=(x_end - 0.2, y_end), xytext=(x_ops[j.num_ops - 1] + 0.2, y),
                    arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))

        if snapshot.phase == 'after':
            if j.job_id in snapshot.done_jobs:
                ax.plot([x_job + 0.2, x_end - 0.2], [y, y_end], '--', color='black', lw=1.0)
            else:
                p = min(snapshot.op_ptr.get(j.job_id, 0), j.num_ops - 1)
                ax.plot([x_job + 0.2, x_ops[p] - 0.2], [y, y], '--', color='black', lw=1.0)
        else:
            if j.job_id not in snapshot.done_jobs:
                p = min(snapshot.op_ptr.get(j.job_id, 0), j.num_ops - 1)
                ax.plot([x_job + 0.2, x_ops[p] - 0.2], [y, y], '--', color='black', lw=1.0)
                for m in j.candidates[p]:
                    ax.plot([x_ops[p], x_m[m]], [y, y_m - 0.1], '--', color=m_colors[m % len(m_colors)], lw=1.0)

        if snapshot.phase == 'after':
            for (jid, oi, mid) in snapshot.selected_edges:
                if jid == j.job_id and oi < j.num_ops:
                    ax.plot([x_ops[oi], x_m[mid]], [y, y_m - 0.1], '-', color=m_colors[mid % len(m_colors)], lw=1.4)

    if state_name:
        ax.set_title(state_name, fontsize=11)
    ax.set_xlim(0, x_end + 0.8)
    ax.set_ylim(0.3, y_m + 0.7)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True
