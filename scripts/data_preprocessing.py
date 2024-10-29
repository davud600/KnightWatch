import pandas as pd
import torch

def load_test_data(filepath):
    data = pd.read_csv(filepath)
    X_test = data.drop(columns=["player_type"])
    y_test = data["player_type"].values
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    return X_test_tensor, y_test_tensor
