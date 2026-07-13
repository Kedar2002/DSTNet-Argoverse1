import torch.nn as nn

from engine.optimizer import (
    build_optimizer,
    optimizer_summary,
)


model = nn.Sequential(

    nn.Linear(32,64),

    nn.LayerNorm(64),

    nn.ReLU(),

    nn.Linear(64,10),

)

optimizer = build_optimizer(
    model=model,
    optimizer="adamw",
    learning_rate=1e-4,
    weight_decay=1e-2,
)

print(optimizer_summary(optimizer))
