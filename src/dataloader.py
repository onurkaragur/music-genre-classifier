from torch.utils.data import DataLoader
from dataset import GTZANDataset

def get_loaders(batch_size=32):
    train_ds = GTZANDataset("data/train")
    val_ds = GTZANDataset("data/val")
    test_ds = GTZANDataset("data/test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader