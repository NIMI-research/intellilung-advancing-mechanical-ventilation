#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List


def run(cmd: List[str], cwd: Path | None = None, env: Dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def parse_root_path(stdout: str) -> str:
    match = re.search(r"Root Path:\s*(.+)", stdout)
    if not match:
        raise RuntimeError(
            "Could not parse root path from create_experiments.py output.\n"
            f"stdout:\n{stdout}"
        )
    return match.group(1).strip()


def compact_indices(indices: List[int]) -> str:
    if not indices:
        return ""
    indices = sorted(set(indices))
    ranges = []
    start = prev = indices[0]

    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = i

    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def get_task_dirs(run_root: Path) -> List[Path]:
    tasks_root = run_root / "tasks"
    if not tasks_root.exists():
        raise RuntimeError(f"Tasks directory does not exist: {tasks_root}")
    return sorted(
        [p for p in tasks_root.iterdir() if p.is_dir()],
        key=lambda p: int(p.name),
    )


def load_task_config(task_dir: Path) -> dict:
    cfg_path = task_dir / "experiment_config.json"
    with open(cfg_path, "r") as f:
        return json.load(f)


def create_experiments(repo_root: Path, config_path: str, device: str) -> Path:
    env = os.environ.copy()
    env["HYPER_PARAM_TUNE_CONFIGS"] = config_path
    env["DEVICE"] = device

    result = run(
        ["python", "hyper_param_tune/create_experiments.py"],
        cwd=repo_root,
        env=env,
    )

    root_path = parse_root_path(result.stdout)
    run_root = Path(root_path)

    if not run_root.exists():
        raise RuntimeError(
            f"Parsed run root does not exist: {run_root}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return run_root


def write_submit_metadata(run_root: Path, config_path: str, num_tasks: int, array_spec: str) -> Path:
    metadata = {
        "run_root": str(run_root),
        "config_path": config_path,
        "num_tasks": num_tasks,
        "array_spec": array_spec,
    }
    metadata_path = run_root / "submit_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def submit_array_job(
    repo_root: Path,
    run_root: Path,
    array_spec: str,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    gpu: int,
    conda_env: str,
    sbatch_script: str,
) -> str:
    cmd = [
        "sbatch",
        f"--partition={partition}",
        f"--array={array_spec}",
        f"--time={time_limit}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        "--gres",
        f"gpu:{gpu}",
        "--export",
        ",".join([
            "ALL",
            f"EXPERIMENT_ROOT={run_root}",
            f"PROJECT_ROOT={repo_root}",
            f"CONDA_ENV_NAME={conda_env}",
        ]),
        sbatch_script,
    ]

    result = run(cmd, cwd=repo_root)
    match = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
    if not match:
        raise RuntimeError(
            "Could not parse job id from sbatch output.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return match.group(1)


def status(run_root: Path) -> None:
    task_dirs = get_task_dirs(run_root)
    total = len(task_dirs)

    finished = []
    pending = []

    for task_dir in task_dirs:
        cfg = load_task_config(task_dir)
        task_id = int(task_dir.name)
        if cfg.get("finished", False):
            finished.append(task_id)
        else:
            pending.append(task_id)

    print(f"Run root:  {run_root}")
    print(f"Total:     {total}")
    print(f"Finished:  {len(finished)}")
    print(f"Pending:   {len(pending)}")
    if pending:
        print(f"Pending IDs: {compact_indices(pending)}")


def resubmit_failed(
    repo_root: Path,
    run_root: Path,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    gpu: int,
    conda_env: str,
    sbatch_script: str,
) -> None:
    task_dirs = get_task_dirs(run_root)
    pending = []

    for task_dir in task_dirs:
        cfg = load_task_config(task_dir)
        if not cfg.get("finished", False):
            pending.append(int(task_dir.name))

    if not pending:
        print("No unfinished tasks found.")
        return

    array_spec = compact_indices(pending)
    job_id = submit_array_job(
        repo_root=repo_root,
        run_root=run_root,
        array_spec=array_spec,
        partition=partition,
        time_limit=time_limit,
        cpus=cpus,
        mem=mem,
        gpu=gpu,
        conda_env=conda_env,
        sbatch_script=sbatch_script,
    )

    print(f"Resubmitted unfinished tasks: {array_spec}")
    print(f"Job ID: {job_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automation wrapper for hyperparameter tuning.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", default=".")
    common.add_argument("--partition", default="alpha")
    common.add_argument("--time", default="3:00:00")
    common.add_argument("--cpus", type=int, default=12)
    common.add_argument("--mem", default="12G")
    common.add_argument("--gpu", type=int, default=1)
    common.add_argument("--conda-env", default="IntelliLungEUEnv")
    common.add_argument("--sbatch-script", default="hyper_param_tune/hparam_array_job.sbatch")

    submit_p = sub.add_parser("submit", parents=[common])
    submit_p.add_argument("config_path")
    submit_p.add_argument("--device", default="cuda")

    status_p = sub.add_parser("status")
    status_p.add_argument("run_root")

    resubmit_p = sub.add_parser("resubmit-failed", parents=[common])
    resubmit_p.add_argument("run_root")

    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()

    if args.cmd == "submit":
        run_root = create_experiments(
            repo_root=repo_root,
            config_path=args.config_path,
            device=args.device,
        )
        task_dirs = get_task_dirs(run_root)
        num_tasks = len(task_dirs)

        if num_tasks == 0:
            raise RuntimeError(f"No tasks created under {run_root / 'tasks'}")

        array_spec = f"0-{num_tasks - 1}"
        write_submit_metadata(run_root, args.config_path, num_tasks, array_spec)

        job_id = submit_array_job(
            repo_root=repo_root,
            run_root=run_root,
            array_spec=array_spec,
            partition=args.partition,
            time_limit=args.time,
            cpus=args.cpus,
            mem=args.mem,
            gpu=args.gpu,
            conda_env=args.conda_env,
            sbatch_script=args.sbatch_script,
        )

        print(f"Run root: {run_root}")
        print(f"Tasks: {num_tasks}")
        print(f"Job ID: {job_id}")
        print(f"Status command:")
        print(f"  python hyper_param_tune/hparam_submit.py status {run_root}")
        print(f"Slurm info:")
        print(f"  sacct -j {job_id} --format=JobID,JobName,State,Elapsed,MaxRSS")

    elif args.cmd == "status":
        status(Path(args.run_root))

    elif args.cmd == "resubmit-failed":
        resubmit_failed(
            repo_root=repo_root,
            run_root=Path(args.run_root),
            partition=args.partition,
            time_limit=args.time,
            cpus=args.cpus,
            mem=args.mem,
            gpu=args.gpu,
            conda_env=args.conda_env,
            sbatch_script=args.sbatch_script,
        )


if __name__ == "__main__":
    main()