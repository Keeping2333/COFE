"""
Experiment-level stratified train/test split.

All scripts (train.py, build_kb.py, diagnose.py) must use the same split
to avoid data leakage. This module provides a single source of truth.

Key principles:
- Split at EXPERIMENT level, not trace level
- Same experiment's traces are never in both train and test
- Stratified by root_cause_type for balanced label distribution
- Deterministic (seed=42) for reproducibility
- Matches baseline methodology (sklearn train_test_split)
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def experiment_level_split(
    samples_index: List[Dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """
    Split experiments into train/test sets, stratified by root_cause_type.

    Args:
        samples_index: dataset._samples_index (list of dicts with run_id/scenario_name + root_cause)
        test_ratio: fraction of experiments for test
        seed: random seed (must match across all scripts)

    Returns:
        (train_experiment_ids, test_experiment_ids)
    """
    from sklearn.model_selection import train_test_split

    # Auto-detect experiment key
    if "run_id" in samples_index[0]:
        exp_key = "run_id"
    else:
        exp_key = "scenario_name"

    # Get unique experiments with their labels
    exp_to_label = {}
    for info in samples_index:
        exp_id = info[exp_key]
        exp_to_label[exp_id] = info["root_cause"]

    experiments = sorted(exp_to_label.keys())  # sort for determinism
    labels = [exp_to_label[e] for e in experiments]

    train_exps, test_exps = train_test_split(
        experiments,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )

    train_set, test_set = set(train_exps), set(test_exps)

    # Print split summary
    train_labels = defaultdict(int)
    test_labels = defaultdict(int)
    for e in train_exps:
        train_labels[exp_to_label[e]] += 1
    for e in test_exps:
        test_labels[exp_to_label[e]] += 1

    print(f"  Experiment-level split (seed={seed}): "
          f"{len(train_set)} train / {len(test_set)} test experiments")
    for ft in sorted(set(labels)):
        print(f"    {ft:35s}: {train_labels[ft]:3d} train / {test_labels[ft]:3d} test")

    return train_set, test_set


def get_experiment_indices(
    samples_index: List[Dict],
    experiment_ids: Set[str],
) -> List[int]:
    """Return trace-level indices belonging to the given experiments."""
    if "run_id" in samples_index[0]:
        exp_key = "run_id"
    else:
        exp_key = "scenario_name"

    return [
        i for i, info in enumerate(samples_index)
        if info[exp_key] in experiment_ids
    ]


def get_one_trace_per_experiment(
    samples_index: List[Dict],
    experiment_ids: Set[str],
) -> List[int]:
    """
    Return one representative trace index per experiment.

    Selects the first trace for each experiment (deterministic).
    Used by diagnose.py for experiment-level evaluation.
    """
    if "run_id" in samples_index[0]:
        exp_key = "run_id"
    else:
        exp_key = "scenario_name"

    seen = set()
    indices = []
    for i, info in enumerate(samples_index):
        exp_id = info[exp_key]
        if exp_id in experiment_ids and exp_id not in seen:
            seen.add(exp_id)
            indices.append(i)
    return indices
