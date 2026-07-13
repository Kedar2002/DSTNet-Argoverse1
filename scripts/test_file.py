import torch

from losses.proposal_loss import ProposalLoss
from models.model_types import Prediction

B = 2
N = 16
K = 6
T = 30

prediction = Prediction(
    trajectories=torch.randn(B, N, K, T, 2),
    scores=torch.randn(B, N, K),
)

gt = torch.randn(B, N, T, 2)

loss = ProposalLoss()

value = loss(
    prediction,
    gt,
)

print(value)
