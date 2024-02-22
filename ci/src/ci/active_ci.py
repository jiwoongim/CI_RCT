from dataclasses import dataclass
from typing import Union

import numpy as np

from ci.util_stats import get_confidence_interval, get_standard_error

PO = 0
SE = 1


@dataclass
class DATA_IND:
    COVARIATE = 0
    ACTION = 1
    OUTCOME = 2
    PROPENSITY = 3


class ActiveCausalInference:
    def __init__(self, num_action: int):
        self._dataset = []
        self.num_action = num_action

    def append_data(
        self, covariate: float, action: float, outcome: float, propensity: float
    ):
        self._dataset.append([covariate, action, outcome, propensity])

    def get_dataset(self) -> np.array:
        return np.asarray(self._dataset)

    def get_empirical_prob_actions(self) -> tuple[float, float]:

        prob_a_list = []
        dataset_np = np.asarray(self._dataset)
        for action in range(self.num_action):
            ind = np.argwhere(dataset_np[:, DATA_IND.ACTION] == action).flatten()
            prob_a_list.append(len(ind) / len(dataset_np))
        return prob_a_list

    def get_potential_outcomes_given_action(self, action: int) -> tuple[np.array, np.array]:

        if self._dataset == []:
            return np.asarray([np.nan]), np.asarray([np.nan])

        dataset_np = np.asarray(self._dataset)
        action_indicators = dataset_np[:, DATA_IND.ACTION] == action
        if len(action_indicators) == 0:
            return np.asarray([np.nan]), np.asarray([np.nan])

        outcomes = dataset_np[:, DATA_IND.OUTCOME] * action_indicators
        potential_outcomes = outcomes/ dataset_np[:, DATA_IND.PROPENSITY]
        return potential_outcomes, outcomes

    def compute_potential_outcome(self, action: int) -> tuple[list[float], float]:
        """Returns potential outcome and conditional outcome"""

        potential_outcomes, outcomes = self.get_potential_outcomes_given_action(action)
        standard_error = get_standard_error(potential_outcomes)
        return [np.nanmean(potential_outcomes), standard_error], np.nanmean(outcomes)

    def compute_potential_outcomes(self):

        conditional_outcome_list = []
        potential_outcome_list = []
        for action in range(self.num_action):
            [potential_outcome, se], conditional_outcome = self.compute_potential_outcome(action)
            potential_outcome_list.append([potential_outcome, se])
            conditional_outcome_list.append(conditional_outcome)

        return np.asarray(potential_outcome_list), np.asarray(conditional_outcome_list)

    def compute_ate(self, treatment_ind: int) -> float:
        potential_outcome_np, _ = self.compute_potential_outcomes()
        argind = np.argsort(potential_outcome_np[:, PO])[::-1]

        ## potential outcome of control is
        ## max_{a!=treatment}potential_outcomes
        po_treatment, se_treatment = potential_outcome_np[treatment_ind]
        if argind[0] == treatment_ind:
            po_control, se_control = potential_outcome_np[argind[1]]
        else:
            po_control, se_control = potential_outcome_np[argind[0]]

        ate = po_treatment - po_control
        se_ate = np.sqrt(se_treatment**2 + se_control**2)
        return [ate, se_ate]
