import torch.nn as nn

from engine.optimizer import build_optimizer
from engine.scheduler import build_scheduler

from engine.checkpoint import (
    save_checkpoint,
    load_checkpoint,
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
    total_steps=100,
)

save_checkpoint(

    path="checkpoint_test.pth",

    model=model,

    optimizer=optimizer,

    scheduler=scheduler,

    epoch=5,

    global_step=200,

    metrics={
        "loss": 1.23,
    },

)

state = load_checkpoint(

    path="checkpoint_test.pth",

    model=model,

    optimizer=optimizer,

    scheduler=scheduler,

)

print(state)
