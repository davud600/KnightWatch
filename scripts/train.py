import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from model import ChessAIModel

class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row.drop("player_type").values.astype(float)
        label = row["player_type"]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for features, label in dataloader:
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    dataset = ChessDataset("data/data.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ChessAIModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs=50)
    torch.save(model.state_dict(), "models/ChessAIModel.pth")
    print("Model saved as 'models/ChessAIModel.pth'")
