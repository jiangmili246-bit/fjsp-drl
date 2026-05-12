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
class GanttSnapshot:
    records: list
    num_machines: int
    disturbance_time: int | None = None


def build_gantt_snapshot(records, num_machines, disturbance_time=None):
    return GanttSnapshot(list(records), int(num_machines), disturbance_time)


def draw_gantt(snapshot: GanttSnapshot, save_path: str, state_name: str = ""):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _fallback_png(save_path)
        return False

    fig, ax = plt.subplots(figsize=(10, 4.8))
    palette = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]

    if snapshot.records:
        for jid, opi, mid, st, en in snapshot.records:
            color = palette[jid % len(palette)]
            ax.barh(mid, en - st, left=st, height=0.7, color=color, edgecolor="black", linewidth=0.8)
            ax.text(st + (en - st) / 2, mid, f"O{jid+1},{opi+1}", ha="center", va="center", fontsize=8)
        xmax = max(r[4] for r in snapshot.records) + 1
    else:
        xmax = 1

    if snapshot.disturbance_time is not None:
        ax.axvline(snapshot.disturbance_time, color="black", linestyle="--", linewidth=1.0)

    if state_name:
        ax.set_title(state_name, fontsize=11)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(list(range(snapshot.num_machines)))
    ax.set_yticklabels([f"M{i+1}" for i in range(snapshot.num_machines)])
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True
