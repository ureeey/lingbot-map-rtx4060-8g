import torch
import torch.nn as nn
from torchvista import trace_model

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear1(x) + self.linear2(x)

model = LinearModel()
model.eval()
example_input = torch.randn(2, 10)

# Visualize the forward pass
trace_model(model, example_input)