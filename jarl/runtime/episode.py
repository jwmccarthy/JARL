import torch as th

from jarl.data.records import EnvStep


class EpisodeTracker:
    def __init__(self) -> None:
        self._reward = None
        self._length = None

    def reset(self) -> None:
        self._reward = None
        self._length = None

    def update(self, step: EnvStep) -> dict[str, dict[str, float]]:
        reward = th.as_tensor(step.reward)
        done = th.as_tensor(step.done, dtype=th.bool, device=reward.device)

        if reward.ndim != 1 or done.shape != reward.shape:
            raise ValueError("episode rewards and done flags must be one-dimensional")
        if self._reward is None:
            self._reward = th.zeros_like(reward)
            self._length = th.zeros_like(reward, dtype=th.int64)

        self._reward += reward
        self._length += 1

        if not done.any():
            return {}

        metrics = {
            "episode": {
                "reward": self._reward[done].mean().item(),
                "length": self._length[done].float().mean().item(),
            },
        }
        self._reward[done] = 0
        self._length[done] = 0

        return metrics
