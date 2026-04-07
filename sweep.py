import itertools
import os
import subprocess
import sys

PYTHON_EXEC = "python3"
MAIN_FILE = "main.py"

# Base command: things that stay fixed
base_cmd = [
    PYTHON_EXEC, MAIN_FILE,
    "cub", "Joint",
    "--seed", "1",
    "-e", "40",
    "-early_stop_patience", "10",
    "-optimizer", "sgd",
    "-momentum", "0.9",
    "-pretrained",
    "-use_attr",
    "-weighted_loss", "multiple",
    "-n_attributes", "112",
    "-attr_loss_weight", "0.001",
    "-normalize_loss",
    "-b", "64",
    "-end2end",
]

# Hyperparameters to test
grid = {
    "lr": [0.01, 0.001, 0.005],
    "weight_decay": [4e-5, 1e-4, 5e-4],
    "scheduler_step": [10, 15],
    "batch_size": [32, 64],
}

# Map grid keys to the command-line flags in your script
flag_map = {
    "lr": "-lr",
    "weight_decay": "-weight_decay",
    "scheduler_step": "-scheduler_step",
    "batch_size": "-b",
}

def build_log_dir(params):
    parts = [
        f"lr_{params['lr']}",
        f"wd_{params['weight_decay']}",
        f"step_{params['scheduler_step']}",
        f"bs_{params['batch_size']}",
    ]
    return "_".join(parts)

def main():
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    experiments = list(itertools.product(*values))

    print(f"Running {len(experiments)} experiments...\n")

    for i, combo in enumerate(experiments, start=1):
        params = dict(zip(keys, combo))
        log_dir = build_log_dir(params)

        cmd = base_cmd + ["-log_dir", log_dir]

        for key in keys:
            cmd += [flag_map[key], str(params[key])]

        print(f"[{i}/{len(experiments)}] Running:")
        print(" ".join(cmd))
        print("-" * 80)

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Experiment failed: {log_dir}")
            # Stop immediately if one run fails
            sys.exit(result.returncode)

    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()