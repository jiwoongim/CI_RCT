from ci.rct import BanditArm, ITEM
from ci.util import sample_bernoulli, RNG

class EpsGreedyBanditArm(BanditArm):
    def __init__(self, num_action: int, param_a:float):
        super().__init__(num_action, param_a)

    # @override
    def sample_action_given_covariate(self, x: float, epsilon:float) -> tuple[int, float]:

        coinflip, _ = sample_bernoulli(epsilon)
        if coinflip:
            return self.sample_action()
        prob = self._action_formula(x)
        return sample_bernoulli(prob)



