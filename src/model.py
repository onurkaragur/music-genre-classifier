import torch.nn as nn

class AudioNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # We'll determine the linear layer size dynamically
        self.flatten = nn.Flatten()
        self.fc1 = None  # Will be set after first forward pass
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        
        # Initialize fc1 on first forward pass if not already done
        if self.fc1 is None:
            fc1_input_size = x.shape[1]
            self.fc1 = nn.Linear(fc1_input_size, 128).to(x.device)
        
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x