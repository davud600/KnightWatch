import torch
from sklearn.metrics import accuracy_score, f1_score
from model import load_model
from data_preprocessing import load_test_data

def load_test_data(filepath):
    import pandas as pd
    data = pd.read_csv(filepath)
    X_test = data.drop(columns=["player_type"])
    y_test = data["player_type"].values
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    return X_test_tensor, y_test_tensor

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze()
        predicted_labels = (predictions >= 0.5).int()

    # calculate metrics with zero_division=1 to handle undefined cases
    accuracy = accuracy_score(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels, zero_division=1)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Model F1 Score: {f1:.2f}")
    print("Predicted labels distribution:", torch.bincount(predicted_labels))

if __name__ == "__main__":
    model = load_model("models/ChessAIModel.pth")
    X_test, y_test = load_test_data("data/evaluation_data.csv")
    
    evaluate_model(model, X_test, y_test)
