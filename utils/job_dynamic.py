import random


def default_due_date(num_ops: int, release_time: float = 0.0, due_date_factor: float = 8.0) -> float:
    return float(release_time + max(1, num_ops) * due_date_factor)


def build_default_job_dynamic(num_jobs: int, nums_ope, due_date_factor: float = 8.0):
    nums_ope = [int(x) for x in nums_ope]
    job_id = list(range(num_jobs))
    rd = [0.0 for _ in range(num_jobs)]
    c = [0 for _ in range(num_jobs)]
    w = [1.0 for _ in range(num_jobs)]
    dd = [default_due_date(nums_ope[i], rd[i], due_date_factor) for i in range(num_jobs)]
    return {
        "job_id": job_id,
        "num_ops": nums_ope,
        "release_time": rd,
        "due_date": dd,
        "arrival_type": c,
        "priority_weight": w,
    }


def generate_dynamic_job_meta(num_jobs: int, nums_ope, scenario: str = "random_arrival", seed: int = None):
    if seed is not None:
        random.seed(seed)
    meta = build_default_job_dynamic(num_jobs, nums_ope)
    horizon = max(50.0, float(sum(int(x) for x in nums_ope)) * 2.0)

    for i in range(num_jobs):
        n_i = int(meta["num_ops"][i])
        base = 6.0 + random.uniform(0.0, 6.0)
        meta["due_date"][i] = default_due_date(n_i, 0.0, base)

    if scenario in ("random_arrival", "concurrent"):
        count = max(1, num_jobs // 5)
        for i in random.sample(range(num_jobs), count):
            rd = random.uniform(1.0, horizon * 0.6)
            meta["release_time"][i] = rd
            meta["arrival_type"][i] = 1
            meta["priority_weight"][i] = 2.0
            n_i = int(meta["num_ops"][i])
            meta["due_date"][i] = default_due_date(n_i, rd, 7.0 + random.uniform(0.0, 5.0))

    if scenario in ("urgent_insertion", "concurrent"):
        count = max(1, num_jobs // 10)
        for i in random.sample(range(num_jobs), count):
            rd = random.uniform(1.0, horizon * 0.4)
            meta["release_time"][i] = rd
            meta["arrival_type"][i] = 2
            meta["priority_weight"][i] = 4.0
            n_i = int(meta["num_ops"][i])
            meta["due_date"][i] = default_due_date(n_i, rd, 4.0 + random.uniform(0.0, 2.0))

    return meta
