import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random


# Determine if CUDA is available and use it for training, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# Definition of the residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Adding the residual value
        return F.relu(out)

class QuixoNet(nn.Module):
    def __init__(self):
        super(QuixoNet, self).__init__()

        self.conv_initial = nn.Conv2d(2, 256, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(256)

        # Creating 10 residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Defining the output layers
        self.fc_val = nn.Linear(256, 1)  # Value head
        self.fc_pol = nn.Linear(256, 44)  # Policy head

    def forward(self, x):
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        x = self.res_blocks(x)
        x = F.avg_pool2d(x, x.size()[2:])  # Global average pooling
        x = x.view(x.size(0), -1)  # Flattening

        # Calculating value and policy predictions
        val = torch.tanh(self.fc_val(x))  # Value prediction
        pol = F.log_softmax(self.fc_pol(x), dim=1)  # Policy prediction

        return pol, val

def train(
    model,
    train_loader,
    optimizer=None,
    criterion_val=None,
    criterion_pol=None,
    epochs=32,
):
    # Initialize optimizer and loss functions if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if criterion_val is None:
        criterion_val = nn.MSELoss()
    if criterion_pol is None:
        criterion_pol = nn.CrossEntropyLoss()

    model.train()
    model = model.to(device)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, val_labels, pol_labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                # Move data to the appropriate device
                inputs = inputs.to(device)
                val_labels = val_labels.to(device)
                pol_labels = pol_labels.to(device)

                # Forward pass, backpropagation, and optimization
                optimizer.zero_grad()
                pol_pred, val_pred = model(inputs)
                loss_val = criterion_val(val_pred.squeeze(), val_labels)
                loss_pol = criterion_pol(pol_pred, pol_labels)
                loss = loss_val + loss_pol
                loss.backward()
                optimizer.step()

                # Update running loss for progress display
                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))


# Define a custom dataset class for Quixo game
class QuixoDataset(Dataset):
    def __init__(self):
        # Initialize lists to store game states and corresponding labels
        self.game_states = []  # List to store game states
        self.val_labels = []  # List to store value labels
        self.pol_labels = []  # List to store policy labels

    # Return the length of the dataset
    def __len__(self):
        return len(self.game_states)

    # Add new data to the dataset
    def add_data(self, new_game_states, new_val_labels, new_pol_labels):
        self.game_states.extend(new_game_states)  # Add new game states
        self.val_labels.extend(new_val_labels)  # Add new value labels
        self.pol_labels.extend(new_pol_labels)  # Add new policy labels

    # Add data and halve the existing dataset randomly
    def add_data_and_halve(self, new_game_states, new_val_labels, new_pol_labels):
        # Halve the existing data randomly
        indices = list(range(len(self.game_states)))
        random.shuffle(indices)
        keep_indices = set(indices[: len(indices) // 2])

        self.game_states = [self.game_states[i] for i in keep_indices]
        self.val_labels = [self.val_labels[i] for i in keep_indices]
        self.pol_labels = [self.pol_labels[i] for i in keep_indices]

        self.add_data(new_game_states, new_val_labels, new_pol_labels)

    # Add data and keep a fixed number of existing data randomly
    def add_data_and_keep_fixed(self, new_game_states, new_val_labels, new_pol_labels, fixed_num):
        # Keep a fixed number of existing data randomly
        if len(self.game_states) > fixed_num:
            indices = list(range(len(self.game_states)))
            random.shuffle(indices)
            keep_indices = set(indices[:fixed_num])

            self.game_states = [self.game_states[i] for i in keep_indices]
            self.val_labels = [self.val_labels[i] for i in keep_indices]
            self.pol_labels = [self.pol_labels[i] for i in keep_indices]

        self.add_data(new_game_states, new_val_labels, new_pol_labels)


    # Retrieve an item from the dataset by index
    def __getitem__(self, idx):
        # Access the game state and labels at the specified index
        state = self.game_states[idx]  # Get the game state
        val_label = self.val_labels[idx]  # Get the value label
        pol_label = self.pol_labels[idx]  # Get the policy label

        # Convert them to tensors before returning
        return (
            torch.tensor(state, dtype=torch.float),  # Game state tensor
            torch.tensor(val_label, dtype=torch.float),  # Value label tensor
            torch.tensor(pol_label, dtype=torch.float),  # Policy label tensor
        )


# Function to create a DataLoader for the Quixo dataset
def get_data_loader(dataset, batch_size, shuffle=True):
    # Create and return a DataLoader with the given dataset and batch size
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
