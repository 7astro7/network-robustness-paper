"""
Main entry point for network robustness experiments.
"""

from runner.run_experiment import run_default


if __name__ == "__main__":
    for seed in [0, 1, 2, 3, 4]:
        print(f"\nRunning seed {seed}")
        q_warn, q_collapse = run_default(seed=seed)
        print(f"q_warn={q_warn}, q_collapse={q_collapse}")
