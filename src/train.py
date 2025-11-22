import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import AudioNN
from dataloader import get_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for mel_specs, labels in tqdm(train_loader, desc="Training"):
        mel_specs = mel_specs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(mel_specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mel_specs, labels in tqdm(val_loader, desc="Validating"):
            mel_specs = mel_specs.to(device)
            labels = labels.to(device)
            
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_epochs = 40
    batch_size = 32
    learning_rate = 0.001
    
    # Get data loaders
    train_loader, val_loader, _ = get_loaders(batch_size=batch_size)
    
    # Create model
    model = AudioNN(num_classes=10)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = "model.pth"
    log_file = "training_log.txt"
    
    # Clear previous log
    with open(log_file, 'w') as f:
        f.write("Training Log\n")
        f.write("="*50 + "\n")
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")
    print(f"Progress will be logged to {log_file}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}\n")
            f.write(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%\n")
            f.write(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"[SAVED] New best model (Val Acc: {val_acc*100:.2f}%)")
            with open(log_file, 'a') as f:
                f.write(f"  [SAVED] New best model (Val Acc: {val_acc*100:.2f}%)\n")
        
        with open(log_file, 'a') as f:
            f.write("\n")
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {best_model_path}")

#python src/train.py to train the model.

