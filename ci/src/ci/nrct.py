import numpy as np
from ci.rct import BanditArm, ITEM
from ci.util import sample_bernoulli, RNG

class NRCTBanditArm(BanditArm):
    def __init__(self, num_action: int, param_a:float):
        super().__init__(num_action, param_a)

    # @override
    def sample_action_given_covariate(self, x: float) -> tuple[int, float]:
        prob = self._action_formula(x)
        return sample_bernoulli(prob)

if __name__ == '__main__':
    num_samples = 100
    bandit = NRCTBanditArm(num_action=2, param_a=0.5)
    covariates = [bandit.sample_covariate() for _ in range(num_samples)]
    prior_actions = [bandit.sample_action()[ITEM] for _ in range(num_samples)]
    rct_outcomes = [bandit.sample_outcome(covariate, action) for (covariate, action) in zip(covariates, prior_actions)]
    rct_outcomes_np = np.asarray(rct_outcomes)

    propensity_actions  = [bandit.sample_action_given_covariate(x)[ITEM] for x in covariates]
    nrct_outcomes = [bandit.sample_outcome(covariate, action) for (covariate, action) in zip(covariates, propensity_actions)]
    nrct_outcomes_np = np.asarray(nrct_outcomes)

    import matplotlib 
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    action1_x1 = np.argwhere((np.asarray(propensity_actions)==1) & (np.asarray(covariates)==1)).flatten()
    action1_x0 = np.argwhere((np.asarray(propensity_actions)==1) & (np.asarray(covariates)==0)).flatten()
    action0_x1 = np.argwhere((np.asarray(propensity_actions)==0) & (np.asarray(covariates)==1)).flatten()
    action0_x0 = np.argwhere((np.asarray(propensity_actions)==0) & (np.asarray(covariates)==0)).flatten()

    axs[0].bar([0,1,2,3], [len(action0_x0), len(action0_x1), len(action1_x0), len(action1_x1)], \
                    label=['Action=0, X=0','Action=0, X=1','Action=1, X=0','Action=1, X=1'], color=['skyblue', 'blue','tomato','indianred'])
    axs[0].set_title("Samples from P(A=a)")
    axs[0].set_ylabel("Frequency of A=a ")
    axs[0].legend()

    def get_density(data):
        density = gaussian_kde(data)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        return density

    xs = np.linspace(0, 1, 20)
    axs[1].set_title("Kernel Estimation, Samples ~ P(Y|A=a,X)")
    axs[1].plot(xs,get_density(rct_outcomes_np[action1_x1])(xs), ls='-', label='Action=1,X=1', color='tomato')
    axs[1].plot(xs,get_density(rct_outcomes_np[action1_x0])(xs), ls='-', label='Action=0,X=0', color='indianred')
    axs[1].plot(xs,get_density(rct_outcomes_np[action0_x1])(xs), ls='-', label='Action=0,X=1', color='blue')
    axs[1].plot(xs,get_density(rct_outcomes_np[action0_x0])(xs), ls='-', label='Action=0,X=0', color='skyblue')
    axs[1].legend()


    plt.tight_layout()
    plt.savefig(f"data_nrct.pdf")