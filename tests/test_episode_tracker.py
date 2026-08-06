import torch as th

from jarl.data.records import EnvStep
from jarl.runtime.episode import EpisodeTracker


def step(reward, done) -> EnvStep:
    reward = th.tensor(reward, dtype=th.float32)
    done = th.tensor(done, dtype=th.bool)
    return EnvStep(None, None, reward, done, th.zeros_like(done))


def test_tracks_completed_episodes() -> None:
    tracker = EpisodeTracker()

    assert tracker.update(step((1, 2), (False, False))) == {}
    assert tracker.update(step((3, 4), (True, False))) == {
        "episode": {"reward": 4.0, "length": 2.0},
    }
    assert tracker.update(step((5, 6), (False, True))) == {
        "episode": {"reward": 12.0, "length": 3.0},
    }
