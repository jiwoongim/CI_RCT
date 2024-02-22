import numpy as np

from ci.active_ci import ActiveCausalInference
from ci import BanditArmABC
from ci.util import sigmoid, sample_bernoulli, RNG

ITEM = 0

class BanditArm(ActiveCausalInference, BanditArmABC):
    def __init__(self, num_action: int, param_a:float):
        super().__init__(num_action)
        self.param_a = param_a
        self.prob_a = self._action_formula(self.param_a)
        print(f"Prob A = {self.prob_a}")

    def sample_covariate(
        self, eps_x: float = 0.7, mode_means: tuple[float, float] = (0.25, 0.75), num_samples:int=1
    ):
        return RNG.binomial(n=1, p=eps_x, size=num_samples).flatten()[ITEM]
        # sample_mode = RNG.binomial(n=1, p=eps_x, size=1).flatten()[ITEM]
        # if sample_mode == 0:
        #    return RNG.normal(mode_means[0], 1, size=1).flatten()[ITEM]
        # else:
        #    return RNG.normal(mode_means[1], 1, size=1).flatten()[ITEM]

    def sample_action(self):
        return sample_bernoulli(self.prob_a)

    def sample_action_given_covariate(self, x: float):
        return self.sample_action()

    def _action_formula(self, x: float):
        eps_a = RNG.normal(0.25, 0.01, size=1).flatten()[ITEM]
        prob = 0.2 * x + np.sqrt(1 - 0.2**2) * eps_a
        assert 0 <= prob <= 1, f"not a probability, {prob}"
        return prob

    def sample_outcome(self, x: float, a: float):
        #eps_y = RNG.normal(0, 0.05, size=1).flatten()[ITEM]
        #return sigmoid(0.1 * a - 0.5 * x + np.sqrt(1 - 0.1**2 - 0.5**2 + 0.02) * eps_y)
        return sigmoid(0.5 * a - 0.1 * x + np.sqrt(1 - 0.1**2 - 0.5**2 + 0.02))


'''
if __name__ == '__main__':
    num_samples = 100
    bandit = BanditArm(num_action=2, param_a=0.5)
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
    action1_x1 = np.argwhere((np.asarray(prior_actions)==1) & (np.asarray(covariates)==1)).flatten()
    action1_x0 = np.argwhere((np.asarray(prior_actions)==1) & (np.asarray(covariates)==0)).flatten()
    action0_x1 = np.argwhere((np.asarray(prior_actions)==0) & (np.asarray(covariates)==1)).flatten()
    action0_x0 = np.argwhere((np.asarray(prior_actions)==0) & (np.asarray(covariates)==0)).flatten()

    axs[0].bar([0,1,2,3], [len(action0_x0), len(action0_x1), len(action1_x0), len(action1_x1)], \
                    label=['Action=0, X=0','Action=0, X=1','Action=1, X=0','Action=1, X=1'], color=['skyblue', 'blue','tomato','red'])
    axs[0].set_title("Samples from P(A=a)")
    axs[0].set_ylabel("Frequency of A=a ")
    axs[0].legend()

    def get_density(data):
        density = gaussian_kde(data)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        return density

    xs = np.linspace(0, 1, 20)
    axs[1].set_title("Kernel Estimation of Empircal Samples ~ P(Y|A=a,X)")
    axs[1].plot(xs,get_density(rct_outcomes_np[action1_x1])(xs), ls='-', label='Action=1,X=1', color='skyblue')
    axs[1].plot(xs,get_density(rct_outcomes_np[action1_x0])(xs), ls='-', label='Action=0,X=0', color='blue')
    axs[1].plot(xs,get_density(rct_outcomes_np[action0_x1])(xs), ls='-', label='Action=0,X=1', color='tomato')
    axs[1].plot(xs,get_density(rct_outcomes_np[action0_x0])(xs), ls='-', label='Action=0,X=0', color='red')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"data_rct.pdf")
'''