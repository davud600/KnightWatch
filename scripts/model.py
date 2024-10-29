import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessAIModel(nn.Module):
    def __init__(self):
        super(ChessAIModel, self).__init__()
        self.fc1 = nn.Linear(138, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.output(x))

def load_model(filepath):
    model = ChessAIModel()
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model
