import torch as th
import torch.nn as nn

from jarl.collect import SelfPlayMatchmaker, SelfPlayRunner, SnapshotPool


def test_live_opponent_snapshots_are_not_evicted() -> None:
    policy = nn.Linear(1, 1)
    pool = SnapshotPool(
        policy,
        max_size=3,
        snapshot_interval=1,
        checkpoint_dir=None,
    )
    pool.add(policy, timesteps=1, protected_ids=(0,))
    pool.add(policy, timesteps=2, protected_ids=(0, 1))

    matchmaker = SelfPlayMatchmaker(
        num_matches=2,
        team_sizes=(1, 1),
        current_fraction=0.5,
        historical_ids=(1,),
        device="cpu",
    )
    matchmaker.opponent_ids[:] = th.tensor((0, -1, 1, -1))
    runner = object.__new__(SelfPlayRunner)
    runner.opponent_pool = pool
    runner.snapshot_policy = policy
    runner.matchmaker = matchmaker
    runner.historical_policies = 1

    runner.after_update(timesteps=3)

    assert 0 in pool.ids
    assert 2 not in pool.ids
    assert pool.policy(0, "cpu") is not None
