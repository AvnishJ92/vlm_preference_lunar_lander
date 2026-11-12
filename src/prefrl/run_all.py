"""Every (condition, seed) run of the config, a few at a time.

    python -m prefrl.run_all --config <run>/config/default.yaml --out <run>/outputs --logs <run>/attempts/001/logs

Runs that need the VLM share the one GPU, so they get their own, smaller pool. A run that already has a summary.json
is skipped; a directory left by an unfinished run is set aside (never appended to) and the run starts over.
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prefrl.train import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--gpu-parallel", type=int, default=2, help="concurrent runs that load the VLM (~20 GB each)")
    parser.add_argument("--cpu-parallel", type=int, default=4, help="concurrent runs that never touch the GPU")
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.logs.mkdir(parents=True, exist_ok=True)

    def one(condition, seed):
        out = args.out / condition / f"seed_{seed}"
        if (out / "summary.json").exists():
            return 0
        if out.exists():
            out.rename(out.with_name(f"{out.name}.incomplete-{int(time.time())}"))
        command = [sys.executable, "-m", "prefrl.train", "--config", args.config,
                   "--condition", condition, "--seed", str(seed), "--out", str(out)]  # fmt: skip
        with (args.logs / f"{condition}_seed{seed}.log").open("w") as log:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode

    with ThreadPoolExecutor(args.gpu_parallel) as gpu, ThreadPoolExecutor(args.cpu_parallel) as cpu:
        jobs = {}
        for condition, spec in cfg["conditions"].items():
            pool = gpu if spec.get("oracle_fraction", 1) < 1 else cpu
            for seed in cfg["seeds"]:
                jobs[condition, seed] = pool.submit(one, condition, seed)
        failed = [job for job, future in jobs.items() if future.result() != 0]
    done = sum((args.out / c / f"seed_{s}" / "summary.json").exists() for c, s in jobs)
    print(f"{done}/{len(jobs)} runs have a summary.json; failed: {failed or 'none'}")
    sys.exit(1 if failed or done < len(jobs) else 0)


if __name__ == "__main__":
    main()
