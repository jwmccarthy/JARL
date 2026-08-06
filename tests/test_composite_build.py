import gymnasium as gym
import torch as th

from jarl.envs.space import torch_space
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy


class TestEnv:
    obs_space = torch_space(gym.spaces.Box(-1, 1, shape=(6,)))
    act_space = torch_space(gym.spaces.MultiDiscrete([3, 2]))


def test_recurrent_policy_builds_children() -> None:
    policy = MultiCategoricalPolicy(
        foot=LinearEncoder(8),
        body=GRU(hidden_size=8),
        head=MLP(dims=[4]),
    ).build(TestEnv())

    output = policy.act(th.zeros(2, 6), policy.initial_state(2))

    assert policy.built
    assert policy.foot.built
    assert policy.body.built
    assert policy.head.built
    assert output.action.shape == (2, 2)
    assert not any(key.startswith("model.") for key in policy.state_dict())


def test_recurrent_critic_builds_children() -> None:
    critic = Critic(
        foot=LinearEncoder(8),
        body=GRU(hidden_size=8),
        head=MLP(dims=[4]),
    ).build(TestEnv())

    value = critic.value(th.zeros(2, 6), critic.initial_state(2))

    assert critic.built
    assert critic.foot.built
    assert critic.body.built
    assert critic.head.built
    assert value.shape == (2,)
