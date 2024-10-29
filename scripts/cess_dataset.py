import torch
from torch.utils.data import Dataset
import pandas as pd

class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row.drop(["player_type", "move_played"]).values.astype(float)
        label = row["player_type"]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
