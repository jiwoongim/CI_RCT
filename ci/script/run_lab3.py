"""python script/rct_script.py --dataset_name online_classroom"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import pandas as pd

from ci.util import get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="online_classroom", help="Dataset File Name"
    )
    return parser.parse_args()


def main(args):
    dataset = get_dataset(args.dataset_name)

    print("Computing Treatments Stats")
    dataset.get_treatment_stats()

    print("Computing Potential Outcome")
    potential_outcomes = dataset.get_potential_outcomes_by_treatment()
    print(potential_outcomes)

    print("Computing ATE")
    po_online = potential_outcomes.loc["online"]["falsexam"]
    po_inclass = potential_outcomes.loc["inclass"]["falsexam"]
    ate = po_online - po_inclass
    print(f"... ATE: {ate}")

    online_se, inclass_se = dataset.get_se_of_outcomes_by_treatment()
    online_ci, inclass_ci = dataset.get_ci_of_outcomes_by_treatment()
    print("95% CI for Online:", online_ci)
    print("95% for for Inclass:", inclass_ci)

    ate_ci, _ = dataset.get_ci_of_average_treatment_effect()
    print("95% for for ATE:", ate_ci)


if __name__ == "__main__":
    args = parse_args()
    main(args)
