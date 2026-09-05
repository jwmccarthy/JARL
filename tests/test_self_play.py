import torch as th
import torch.nn as nn

from jarl.collect import SelfPlayMatchmaker, SelfPlayRunner, SnapshotPool


def test_stale_live_opponents_are_remapped_before_the_next_rollout() -> None:
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
    matchmaker.opponent_ids[:] = th.tensor((0, -1, 2, -1))
    runner = object.__new__(SelfPlayRunner)
    runner.opponent_pool = pool
    runner.snapshot_policy = policy
    runner.matchmaker = matchmaker
    runner.historical_policies = 2
    runner.state = th.ones((4, 1, 2))

    runner.after_update(timesteps=3)

    assert pool.ids == (1, 2, 3)
    assert matchmaker.opponent_ids[0].item() in (2, 3)
    assert matchmaker.opponent_ids[1:].tolist() == [-1, 2, -1]
    assert runner.state[0].count_nonzero() == 0
    assert runner.state[1:].eq(1).all()
    assert all(
        pool.policy(snapshot_id, "cpu") is not None
        for snapshot_id in (1, 2, 3)
    )


def test_stale_multi_car_opponents_are_remapped_as_one_team() -> None:
    matchmaker = SelfPlayMatchmaker(
        num_matches=2,
        team_sizes=(2, 1),
        current_fraction=0.5,
        historical_ids=(2, 3),
        device="cpu",
    )
    matchmaker.opponent_ids[:] = th.tensor((0, 0, -1, -1, -1, 2))

    remapped = matchmaker.remap_stale_opponents()

    assert matchmaker.opponent_ids[0].item() in (2, 3)
    assert matchmaker.opponent_ids[1] == matchmaker.opponent_ids[0]
    assert matchmaker.opponent_ids[2:].tolist() == [-1, -1, -1, 2]
    assert remapped.tolist() == [True, True, False, False, False, False]
