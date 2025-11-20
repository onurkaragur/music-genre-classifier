import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from model import AudioNN
from dataloader import get_loaders


def evaluate(model, test_loader, device):
    model.eval()
    total = 0
    correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mel_specs, labels in test_loader:
            mel_specs = mel_specs.to(device)
            labels = labels.to(device)

            outputs = model(mel_specs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_loaders(batch_size=32)
    model = AudioNN(num_classes=10)
    
    model_path = "model.pth"
    try:
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Initialize fc1 by doing a dummy forward pass if needed
        dummy_input = torch.randn(1, 1, 64, 130).to(device)  # Approximate mel spec shape
        _ = model(dummy_input)
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    except Exception as e:
        print(f"Warning: Error loading model: {e}")
        print("Using untrained model.")
    
    model = model.to(device)
    
    test_accuracy, y_pred, y_true = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))