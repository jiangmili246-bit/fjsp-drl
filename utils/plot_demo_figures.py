def _fallback_png(path):
    """Generate a non-blank fallback PNG if matplotlib is unavailable."""
    import struct
    import zlib

    w, h = 480, 320
    row = bytes([255, 255, 255]) * w
    img = [bytearray(row) for _ in range(h)]

    # Draw simple black axes so the fallback is not a 1x1 blank image.
    for x in range(40, w - 20):
        i = x * 3
        img[h - 40][i:i + 3] = b"\x00\x00\x00"

    for y in range(20, h - 40):
        i = 40 * 3
        img[y][i:i + 3] = b"\x00\x00\x00"

    raw = b"".join(b"\x00" + bytes(r) for r in img)
    comp = zlib.compress(raw, 9)

    def chunk(tag, data):
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", comp)
        + chunk(b"IEND", b"")
    )

    with open(path, "wb") as f:
        f.write(png)


def _job_label(job):
    if job.c == 0:
        return rf"$J_{{{job.job_id + 1}}}$"
    if job.c == 1:
        return rf"$J_{{{job.job_id + 1}}}^r$"
    return rf"$J_{{{job.job_id + 1}}}^{{u*}}$"


def plot_hetero_graph_before(path, jobs, initial_jobs, num_m):
    """
    Plot heterogeneous graph before scheduling / before disturbance.
    Semantics:
    - Only initial jobs are shown.
    - J-O dashed edges point to the first active operation.
    - O-M dashed candidate edges are drawn only for first operations.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(12, 6))

        m_y = 0.88
        m_xs = [0.55, 0.70, 0.85]
        job_x = 0.08
        op_xs = [0.28, 0.40, 0.52]
        ys = [0.72, 0.52, 0.32]

        m_colors = ["red", "green", "dodgerblue"]

        # Machine nodes
        for m in range(num_m):
            ax.scatter(
                m_xs[m],
                m_y,
                s=360,
                marker="o",
                color="white",
                edgecolors="black",
                linewidths=1.3,
                zorder=3,
            )
            ax.text(
                m_xs[m],
                m_y,
                rf"$M_{{{m + 1}}}$",
                ha="center",
                va="center",
                fontsize=12,
                fontfamily="serif",
                zorder=4,
            )

        # Initial jobs and operations
        for idx, j in enumerate(initial_jobs):
            y = ys[idx]

            ax.scatter(
                job_x,
                y,
                s=330,
                marker="o",
                color="white",
                edgecolors="black",
                linewidths=1.3,
                zorder=3,
            )
            ax.text(
                job_x,
                y,
                _job_label(j),
                ha="center",
                va="center",
                fontsize=12,
                fontfamily="serif",
                zorder=4,
            )

            for oi in range(j.num_ops):
                fill = "#F9E79F" if oi == 0 else "white"
                ox = op_xs[oi]

                ax.scatter(
                    ox,
                    y,
                    s=420,
                    marker="o",
                    color=fill,
                    edgecolors="black",
                    linewidths=1.3,
                    zorder=3,
                )
                ax.text(
                    ox,
                    y,
                    rf"$O_{{{j.job_id + 1},{oi + 1}}}$",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontfamily="serif",
                    zorder=4,
                )

                if oi < j.num_ops - 1:
                    ax.annotate(
                        "",
                        xy=(op_xs[oi + 1] - 0.025, y),
                        xytext=(ox + 0.025, y),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
                    )

            # J-O active edge
            ax.plot(
                [job_x + 0.025, op_xs[0] - 0.03],
                [y, y],
                "--",
                color="black",
                lw=1.1,
            )

            # O-M candidate edges for the first operation only
            for m in j.candidates[0]:
                ax.plot(
                    [op_xs[0] + 0.025, m_xs[m] - 0.02],
                    [y + 0.015, m_y - 0.025],
                    "--",
                    color=m_colors[m],
                    lw=1.2,
                )

        # End node shown without completion edges in before graph
        ax.scatter(
            0.95,
            0.52,
            s=380,
            marker="o",
            color="white",
            edgecolors="black",
            linewidths=1.3,
            zorder=3,
        )
        ax.text(
            0.95,
            0.52,
            r"$End$",
            ha="center",
            va="center",
            fontsize=12,
            fontfamily="serif",
            zorder=4,
        )

        legend = [
            Line2D([0], [0], marker="o", color="w", label="Job node",
                   markerfacecolor="white", markeredgecolor="black", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Active operation",
                   markerfacecolor="#F9E79F", markeredgecolor="black", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Non-active operation",
                   markerfacecolor="white", markeredgecolor="black", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Machine node",
                   markerfacecolor="white", markeredgecolor="black", markersize=10),
            Line2D([0], [0], linestyle="-", color="black", label="Operation sequence"),
            Line2D([0], [0], linestyle="--", color="black", label="J-O active edge"),
            Line2D([0], [0], linestyle="--", color="red", label="O-M candidate M1"),
            Line2D([0], [0], linestyle="--", color="green", label="O-M candidate M2"),
            Line2D([0], [0], linestyle="--", color="dodgerblue", label="O-M candidate M3"),
        ]

        ax.legend(handles=legend, loc="lower right", fontsize=8)
        ax.set_title("Heterogeneous Graph Before Scheduling", fontsize=15, fontfamily="serif")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.18, 0.98)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return True

    except Exception as e:
        print(f"Plot failed for {path}: {e}")
        _fallback_png(path)
        return False


def plot_hetero_graph_after(path, jobs, schedule_records, num_m):
    """
    Plot heterogeneous graph after scheduling.
    Semantics:
    - All jobs are shown.
    - All operations are completed.
    - Selected O-M edges are drawn from schedule_records.
    - Completion edges connect last operations to End.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(13, 8))

        m_y = 0.92
        m_xs = [0.50, 0.68, 0.86]
        job_x = 0.08
        op_xs = [0.25, 0.39, 0.53]
        y_start, y_gap = 0.84, 0.11
        end_x, end_y = 0.93, 0.50

        m_colors = ["red", "green", "dodgerblue"]
        sel_edges = {(jid, opi, mid) for (jid, opi, mid, _, _) in schedule_records}

        # Machine nodes
        for m in range(num_m):
            ax.scatter(
                m_xs[m],
                m_y,
                s=360,
                marker="o",
                color="white",
                edgecolors="black",
                linewidths=1.3,
                zorder=3,
            )
            ax.text(
                m_xs[m],
                m_y,
                rf"$M_{{{m + 1}}}$",
                ha="center",
                va="center",
                fontsize=12,
                fontfamily="serif",
                zorder=4,
            )

        # Jobs and operations
        for idx, j in enumerate(sorted(jobs, key=lambda x: x.job_id)):
            y = y_start - idx * y_gap

            if j.c == 0:
                job_face = "white"
                job_edge = "black"
                job_color = "black"
            elif j.c == 1:
                job_face = "#EAF4FF"
                job_edge = "dodgerblue"
                job_color = "dodgerblue"
            else:
                job_face = "#FFF0F0"
                job_edge = "red"
                job_color = "red"

            ax.scatter(
                job_x,
                y,
                s=330,
                marker="o",
                color=job_face,
                edgecolors=job_edge,
                linewidths=1.6,
                zorder=3,
            )
            ax.text(
                job_x,
                y,
                _job_label(j),
                ha="center",
                va="center",
                fontsize=12,
                fontfamily="serif",
                color=job_color,
                zorder=4,
            )

            if j.c == 2:
                ax.text(
                    job_x + 0.035,
                    y - 0.035,
                    r"$w=4$",
                    color="red",
                    fontsize=10,
                    fontfamily="serif",
                )

            for oi in range(j.num_ops):
                ox = op_xs[oi]

                ax.scatter(
                    ox,
                    y,
                    s=400,
                    marker="o",
                    color="#E5E7E9",
                    edgecolors="black",
                    linewidths=1.2,
                    zorder=3,
                )
                ax.text(
                    ox,
                    y,
                    rf"$O_{{{j.job_id + 1},{oi + 1}}}$",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontfamily="serif",
                    zorder=4,
                )

                if oi < j.num_ops - 1:
                    ax.annotate(
                        "",
                        xy=(op_xs[oi + 1] - 0.022, y),
                        xytext=(ox + 0.022, y),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                    )

                # Selected O-M solid edge
                for m in range(num_m):
                    if (j.job_id, oi, m) in sel_edges:
                        ax.plot(
                            [m_xs[m], ox],
                            [m_y - 0.025, y + 0.025],
                            "-",
                            color=m_colors[m],
                            lw=1.3,
                            alpha=0.72,
                        )

            # Completion edge: last operation to End only
            ax.plot(
                [op_xs[-1] + 0.025, end_x - 0.025],
                [y, end_y],
                "-",
                color="black",
                lw=0.9,
                alpha=0.65,
            )

        # End node
        ax.scatter(
            end_x,
            end_y,
            s=400,
            marker="o",
            color="white",
            edgecolors="black",
            linewidths=1.3,
            zorder=3,
        )
        ax.text(
            end_x,
            end_y,
            r"$End$",
            ha="center",
            va="center",
            fontsize=12,
            fontfamily="serif",
            zorder=4,
        )

        legend = [
            Line2D([0], [0], marker="o", color="w", label="Initial job",
                   markerfacecolor="white", markeredgecolor="black", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Random arrival job",
                   markerfacecolor="#EAF4FF", markeredgecolor="dodgerblue", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Urgent insertion job",
                   markerfacecolor="#FFF0F0", markeredgecolor="red", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Completed operation",
                   markerfacecolor="#E5E7E9", markeredgecolor="black", markersize=10),
            Line2D([0], [0], linestyle="-", color="black", label="Operation sequence / completion"),
            Line2D([0], [0], linestyle="-", color="red", label="Selected O-M from M1"),
            Line2D([0], [0], linestyle="-", color="green", label="Selected O-M from M2"),
            Line2D([0], [0], linestyle="-", color="dodgerblue", label="Selected O-M from M3"),
        ]

        ax.legend(handles=legend, loc="lower right", fontsize=8)
        ax.set_title("Heterogeneous Graph After Scheduling", fontsize=15, fontfamily="serif")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.1, 0.98)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return True

    except Exception as e:
        print(f"Plot failed for {path}: {e}")
        _fallback_png(path)
        return False


def plot_gantt(path, title, schedule_records, job_map):
    """Plot before/after Gantt chart."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(11, 5))

        if not schedule_records:
            for m in range(3):
                ax.barh(
                    m,
                    1,
                    left=0,
                    color="white",
                    edgecolor="black",
                    linewidth=1.0,
                )
            ax.text(
                0.5,
                1.0,
                "Initial waiting-state\n(no assigned operations yet)",
                fontsize=11,
                ha="center",
                va="center",
                fontfamily="serif",
            )
        else:
            cmap = {
                0: "#95A5A6",
                1: "#3498DB",
                2: "#E74C3C",
            }

            for jid, opi, m, st, en in schedule_records:
                j = job_map[jid]
                lw = 2.2 if j.c == 2 else 1.0

                ax.barh(
                    m,
                    en - st,
                    left=st,
                    color=cmap[j.c],
                    edgecolor="black",
                    linewidth=lw,
                )
                ax.text(
                    st + (en - st) / 2,
                    m,
                    rf"$O_{{{jid + 1},{opi + 1}}}$",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontfamily="serif",
                )

        ax.set_title(title, fontsize=14, fontfamily="serif")
        ax.set_xlabel("Time", fontsize=12, fontfamily="serif")
        ax.set_ylabel("Machine", fontsize=12, fontfamily="serif")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([r"$M_1$", r"$M_2$", r"$M_3$"], fontfamily="serif")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        handles = [
            Patch(facecolor="#95A5A6", edgecolor="black", label="Initial Job"),
            Patch(facecolor="#3498DB", edgecolor="black", label="Random Arrival Job"),
            Patch(facecolor="#E74C3C", edgecolor="black", label="Urgent Insertion Job"),
        ]

        ax.legend(handles=handles, loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return True

    except Exception as e:
        print(f"Plot failed for {path}: {e}")
        _fallback_png(path)
        return False