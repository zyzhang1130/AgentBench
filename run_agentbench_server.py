#!/usr/bin/env python
"""Simple launcher for AgentBench task server.

This script starts a task_controller and a single task_worker on the
specified port so that the HTTP endpoints used by the training script
are available on localhost.

Example:
    python run_agentbench_server.py --task dbbench-std --port 8000

The task definitions come from ``configs/tasks/task_assembly.yaml`` by
default.  The controller listens on port 5000 unless overridden.
"""

import argparse
import os
import pathlib
import subprocess
import time

REPO = pathlib.Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="Launch AgentBench task server")
    p.add_argument("--task", default="dbbench-std", help="Task name defined in task_assembly.yaml")
    p.add_argument(
        "--config",
        default="configs/tasks/task_assembly.yaml",
        help="Task configuration file for task_worker",
    )
    p.add_argument("--port", type=int, default=8000, help="Port for the task worker")
    p.add_argument(
        "--ctrl-port", type=int, default=5000, help="Port for the task controller"
    )
    args = p.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)

    # 1. start the task_controller
    ctrl_cmd = ["python", "-m", "src.server.task_controller", "--port", str(args.ctrl_port)]
    ctrl_proc = subprocess.Popen(ctrl_cmd, cwd=REPO, env=env)
    time.sleep(3)  # give controller time to come up

    # 2. start the task_worker for the requested task
    worker_cmd = [
        "python",
        "-m",
        "src.server.task_worker",
        args.task,
        "--config",
        args.config,
        "--port",
        str(args.port),
        "--self",
        f"http://localhost:{args.port}/api",
        "--controller",
        f"http://localhost:{args.ctrl_port}/api",
    ]
    worker_proc = subprocess.Popen(worker_cmd, cwd=REPO, env=env)

    print(
        f"AgentBench task server for '{args.task}' running at http://localhost:{args.port}/api"
    )

    try:
        worker_proc.wait()
    finally:
        ctrl_proc.terminate()
        worker_proc.terminate()


if __name__ == "__main__":
    main()
