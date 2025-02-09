import time
import torch as th

from jarl.log.log import Progress


if __name__ == "__main__":
    bar = Progress(530, width=30)
    for _ in bar:
        bar.update(episode=dict(
            mean_rew=th.rand(1),
            mean_len=th.rand(1)
        ))
        bar.update(updates=dict(
            bce_loss=th.rand(1),
            pol_loss=th.rand(1)
        ))
        time.sleep(0.1)