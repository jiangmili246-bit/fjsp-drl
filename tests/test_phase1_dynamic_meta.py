import os
import sys

sys.path.append(os.path.abspath("."))


from utils.dynamic_instance_generator import build_dynamic_instance_from_fjs_file


def check_fields(meta):
    req = ["job_id", "num_ops", "release_time", "due_date", "arrival_type", "priority_weight"]
    for k in req:
        assert k in meta, f"missing field: {k}"
    n = len(meta["job_id"])
    for k in req:
        assert len(meta[k]) == n, f"length mismatch for {k}"


if __name__ == "__main__":
    f = "data_dev/1005/10j_5m_001.fjs"
    assert os.path.exists(f), f"missing sample file: {f}"

    for scenario in ["random_arrival", "urgent_insertion", "concurrent"]:
        inst = build_dynamic_instance_from_fjs_file(f, scenario=scenario, seed=7)
        meta = inst["job_meta"]
        check_fields(meta)
        print(scenario, "OK", "RD sample:", meta["release_time"][:3], "DD sample:", meta["due_date"][:3], "c sample:", meta["arrival_type"][:3], "w sample:", meta["priority_weight"][:3])

    print("Phase1 dynamic meta test passed.")
