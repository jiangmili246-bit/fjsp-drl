import json
import os
from typing import Dict, List

from utils.job_dynamic import generate_dynamic_job_meta


def _parse_num_ops_per_job(lines):
    header = lines[0].strip().split()
    num_jobs = int(header[0])
    num_mas = int(header[1])
    nums_ope = []
    for i in range(1, 1 + num_jobs):
        if i >= len(lines):
            break
        line = lines[i].strip()
        if not line:
            continue
        nums_ope.append(int(line.split()[0]))
    return num_jobs, num_mas, nums_ope


def build_dynamic_instance_from_fjs_file(fjs_path: str, scenario: str = "random_arrival", seed: int = 0) -> Dict:
    with open(fjs_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    num_jobs, num_mas, nums_ope = _parse_num_ops_per_job(lines)
    job_meta = generate_dynamic_job_meta(num_jobs=num_jobs, nums_ope=nums_ope, scenario=scenario, seed=seed)

    return {
        "source_file": fjs_path,
        "scenario": scenario,
        "num_jobs": num_jobs,
        "num_mas": num_mas,
        "job_meta": {
            "job_id": job_meta["job_id"],
            "num_ops": job_meta["num_ops"],
            "release_time": job_meta["release_time"],
            "due_date": job_meta["due_date"],
            "arrival_type": job_meta["arrival_type"],
            "priority_weight": job_meta["priority_weight"],
        },
    }


def generate_dynamic_dataset(fjs_files: List[str], out_dir: str, scenario: str, seed: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    all_instances = []
    for idx, f in enumerate(fjs_files):
        inst = build_dynamic_instance_from_fjs_file(f, scenario=scenario, seed=seed + idx)
        all_instances.append(inst)
        out_path = os.path.join(out_dir, os.path.basename(f) + f".{scenario}.meta.json")
        with open(out_path, "w", encoding="utf-8") as wf:
            json.dump(inst, wf, indent=2, ensure_ascii=False)
    return all_instances


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--scenario", type=str, choices=["random_arrival", "urgent_insertion", "concurrent"], required=True)
    parser.add_argument("--out_dir", type=str, default="./data_dynamic")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = [os.path.join(args.data_dir, x) for x in sorted(os.listdir(args.data_dir)) if x.endswith(".fjs")]
    files = files[:args.limit]
    generate_dynamic_dataset(files, args.out_dir, args.scenario, args.seed)
    print(f"Generated {len(files)} dynamic metadata files in {args.out_dir}")
