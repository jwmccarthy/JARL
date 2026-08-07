import torch as th

from jarl.data.records import EnvStep
from jarl.runtime.episode import EpisodeTracker


def step(reward, done, groups=None) -> EnvStep:
    reward = th.tensor(reward, dtype=th.float32)
    done = th.tensor(done, dtype=th.bool)
    return EnvStep(
        None,
        None,
        reward,
        done,
        th.zeros_like(done),
        episode_groups=groups or {},
    )


def test_tracks_completed_episodes() -> None:
    tracker = EpisodeTracker()

    assert tracker.update(step((1, 2), (False, False))) == {}
    assert tracker.update(step((3, 4), (True, False))) == {
        "episode": {"reward": 4.0, "length": 2.0},
    }
    assert tracker.update(step((5, 6), (False, True))) == {
        "episode": {"reward": 12.0, "length": 3.0},
    }


def test_tracks_named_episode_groups() -> None:
    tracker = EpisodeTracker()
    groups = {
        "current":    th.tensor((True, False)),
        "historical": th.tensor((False, True)),
    }

    assert tracker.update(step((1, 2), (False, False), groups)) == {}
    assert tracker.update(step((3, 4), (True, True), groups)) == {
        "episode": {
            "reward":             5.0,
            "length":             2.0,
            "current_reward":     4.0,
            "current_length":     2.0,
            "historical_reward": 6.0,
            "historical_length": 2.0,
        },
    }
