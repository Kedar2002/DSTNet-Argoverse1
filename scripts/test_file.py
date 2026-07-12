import torch

from models.model_types import Prediction
from models.refinement.refinement import Refinement

B = 2
N = 16
K = 6
T = 30
C = 256

scene = torch.randn(B, N, C)

prediction = Prediction(
    trajectories=torch.randn(B, N, K, T, 2),
    scores=torch.randn(B, N, K),
)

model = Refinement(
    hidden_dim=C,
    prediction_steps=T,
)

output = model(
    encoder_features=scene,
    prediction=prediction,
)

print("Trajectories:", output.trajectories.shape)
print("Scores:", output.scores.shape)
print("Offsets:", output.offsets.shape)
