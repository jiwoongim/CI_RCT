import numpy as np

from ci.active_ci import DATA_IND, PO 
from ci.rct import BanditArm
from ci.util import ITEM, RNG, sample_bernoulli
from ci.util_stats import get_confidence_interval, get_standard_error


class ThompsonBandit(BanditArm):
    def __init__(self, num_action: int, param_a:float):
        super().__init__(num_action, param_a)

    # @override
    def sample_action_given_covariate(self, covariate: float) -> tuple[int, float]:
        """ This function is for covariates with binary values [0,1]^D 
        where D is dimensionality"""

        if self._dataset == []:
            return self.sample_action()
        dataset_np = np.asarray(self._dataset)

        ## Resort to rct if have never seen covariate X=x 
        argind = np.argwhere(covariate == dataset_np[:,DATA_IND.COVARIATE]).flatten()
        if len(argind) == 0:
            return self.sample_action()
        dataset_cov_np = dataset_np[argind]

        ## Compute propensity score P(A|X=x)
        propensity_scores = []
        for action in range(self.num_action):

            ## Make sure at least all actions are explored 
            ## at least once before applying thompson smapling
            ## i.e., there is at least one sample for p(A=a|X=x)
            inds = np.argwhere(
                dataset_cov_np[:, DATA_IND.ACTION] == action
            ).flatten()
            if len(inds) == 0:
                return self.sample_action()

            ## Normalize the outcome by the propensity score
            potential_outcome_list = dataset_cov_np[inds, DATA_IND.OUTCOME] / dataset_cov_np[inds, DATA_IND.PROPENSITY]

            ## Compute the potentials 
            exp_potential_outcome = np.exp(np.nanmean(potential_outcome_list))
            propensity_scores.append(exp_potential_outcome)

        ## P(A|X) follows the Boltzmann distribution
        propensity_scores_np = np.asarray(propensity_scores) / sum(propensity_scores) 
        action, prob = sample_bernoulli(propensity_scores_np[0])
        return action, prob 


