import torch.nn as nn

from engine.optimizer import build_optimizer
from engine.scheduler import (
    build_scheduler,
)

model = nn.Linear(
    32,
    16,
)

optimizer = build_optimizer(
    model,
)

scheduler = build_scheduler(

    optimizer,

    scheduler="warmup_cosine",

    total_steps=10000,

    warmup_steps=1000,

)

print(scheduler)

for i in range(5):

    optimizer.step()

    scheduler.step()

    print(

        optimizer.param_groups[0]["lr"]

    )
