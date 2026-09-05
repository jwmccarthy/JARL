import gymnasium as gym
import torch as th

from jarl.collect import RecurrentCriticCapture
from jarl.collect.capture import CaptureContext
from jarl.data.records import EnvStep, PolicyOutput
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


def test_recurrent_critic_capture_is_concrete() -> None:
    critic = Critic(
        foot=LinearEncoder(8),
        body=GRU(hidden_size=8),
        head=MLP(dims=[4]),
    ).build(TestEnv())

    capture = RecurrentCriticCapture(critic)
    capture.reset(batch_size=2)
    context = CaptureContext(
        observation=th.ones(2, 6),
        state=None,
        policy_output=PolicyOutput(action=th.zeros(2, 2)),
        env_step=EnvStep(
            next_obs=th.full((2, 6), 2.0),
            observation=None,
            reward=th.zeros(2),
            terminated=th.tensor([False, True]),
            truncated=th.zeros(2, dtype=th.bool),
        ),
    )

    result = capture(context)

    assert capture.state.shape == (2, 1, 8)
    assert set(result) == {"critic_state", "baseline_value", "baseline_next_value"}
    assert result["critic_state"].shape == (2, 1, 8)
    assert result["baseline_value"].shape == (2,)
    assert result["baseline_next_value"].shape == (2,)
    assert capture.state[0].abs().sum() > 0
    assert capture.state[1].abs().sum() == 0
