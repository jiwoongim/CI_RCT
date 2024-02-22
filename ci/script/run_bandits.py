"""
ShortCut:
python script/run_bandits.py --algo_type rct --num_run 2000 --num_action 2
python script/run_bandits.py --algo_type thompson --num_run 2000 --num_action 2
"""
import argparse
import os
import sys

import numpy as np

from ci.rct import BanditArm
from ci.nrct import NRCTBanditArm 
from ci.thompson import ThompsonBandit
from ci.eps_greedy import EpsGreedyBanditArm
from ci.util import MEAN, SE, Bookkeeping

RNG = np.random.RandomState()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo_type", type=str, default="rct", help="rct | nrct | eps | thompson"
    )
    parser.add_argument("--num_run", type=int, default=3000)
    parser.add_argument("--num_action", type=int, default=2)
    parser.add_argument("--param_a", type=float, default=0.5)
    return parser.parse_args()


def get_model(args):
    if args.algo_type == "thompson":
        return ThompsonBandit(args.num_action, param_a=args.param_a)
    elif args.algo_type == "nrct":
        return NRCTBanditArm(args.num_action, param_a=args.param_a)
    elif args.algo_type == "epsilon":   
        return EpsGreedyBanditArm(args.num_action, param_a=args.param_a)
    return BanditArm(args.num_action, param_a=args.param_a)


def get_action(args, model, x):
    if args.algo_type == "rct":
        return model.sample_action()
    return model.sample_action_given_covariate(x)


def main(args):
    experiment = Bookkeeping(args.num_action)

    model = get_model(args)
    for i in range(args.num_run):
        x_i = model.sample_covariate()
        a_i, p_i = get_action(args, model, x_i)
        y_i = model.sample_outcome(x_i, a_i)
        model.append_data(x_i, a_i, y_i, p_i)
        prob_actions = model.get_empirical_prob_actions()

        if i < 100:
            continue
        #print(f"Sampled Data: {x_i, a_i, y_i, p_i}, {prob_actions}")


        ## po - potential outcomes 
        ## co - conditional outcomes 
        po0_stats, co0_stats = model.compute_potential_outcome(action=0)
        po1_stats, co1_stats = model.compute_potential_outcome(action=1)
        ate_stat = model.compute_ate(treatment_ind=0)

        if i % 500 == 0:
            print(
                f"...Iteration {i}"
                + f"| EPO0 {po0_stats[MEAN]:.3f} +/- {po0_stats[SE]:.3f}"\
                + f"| EPO1 {po1_stats[MEAN]:.3f} +/- {po1_stats[SE]:.3f}"\
                + f"| PO0 {co0_stats/prob_actions[0]:.3f} PO1 {co1_stats/prob_actions[1]:.3f}"\
                + f"| CO0 {co0_stats:.3f} CO1 {co1_stats:.3f}"\
                + f"| PA0 {prob_actions[0]:.3f} PA1 {prob_actions[1]:.3f}"\
                + f"| ATE {ate_stat[MEAN]:.3f} +/- {ate_stat[SE]:.3f}"
            )
        experiment.add_potential_outcomes(i, [po0_stats, po1_stats])
        experiment.add_conditional_outcomes(i, [co0_stats, co1_stats])
        experiment.add_ate(i, ate_stat)

    fname=f"{args.algo_type}_bandit_{args.num_action}actions_{args.num_run}runs"
    if args.algo_type == 'rct':
        fname += f"_{model.param_a}paramA"
    experiment.save_figure(fname)


if __name__ == "__main__":
    args = parse_args()
    main(args)
