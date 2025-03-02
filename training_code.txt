import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn import init
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(EncoderBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ConvBNReLU(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.encoder(x)
        x, indices = self.pool(x)
        return x, indices

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(DecoderBlock, self).__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(ConvBNReLU(in_channels, in_channels))
        # Adjust the last layer to have the appropriate number of input and output channels
        layers.append(ConvBNReLU(in_channels, out_channels))
        self.decoder = nn.Sequential(*layers)
        self.last_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, indices,is_last_block=False):
        x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        if is_last_block:
                for layer in self.decoder[:-1]:
                        x = layer(x)
                x = self.last_conv(x)
        else:
                x = self.decoder(x)
        return x

class SegNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegNet, self).__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64, 2)
        self.enc2 = EncoderBlock(64, 64, 2)
        self.enc3 = EncoderBlock(64, 128, 3)
        self.enc4 = EncoderBlock(128, 256, 3)
        self.enc5 = EncoderBlock(256, 512, 3)
        
        # Decoder
        self.dec5 = DecoderBlock(512, 256, 3)
        self.dec4 = DecoderBlock(256, 128, 3)
        self.dec3 = DecoderBlock(128, 64, 3)
        self.dec2 = DecoderBlock(64, 64, 2)
        self.dec1 = DecoderBlock(64, num_classes, 2)

    def forward(self, x):
        # Encoder
        x, indices1 = self.enc1(x)
        x, indices2 = self.enc2(x)
        x, indices3 = self.enc3(x)
        x, indices4 = self.enc4(x)
        x, indices5 = self.enc5(x)
        
        # Decoder
        x = self.dec5(x, indices5)
        x = self.dec4(x, indices4)
        x = self.dec3(x, indices3)
        x = self.dec2(x, indices2)
        x = self.dec1(x, indices1,is_last_block=True)
        
        return x








class NYUDataset(Dataset):
    def __init__(self, image_with_depth_path, label_path, transform=None):
        self.images_with_depth =  torch.load( image_with_depth_path)
        self.labels =  torch.load(label_path)

        self.transform = transform  # Optional transformation for data augmentation

    def __len__(self):
        return len(self.images_with_depth)

    def __getitem__(self, idx):
        image_with_depth = self.images_with_depth[idx]
        label = self.labels[idx]

        if self.transform:
            image_with_depth, label = self.transform(image_with_depth, label)

        return image_with_depth, label
    def shuffle_data(self):
        """
        Shuffle the data samples.
        """
        indices = torch.randperm(len(self.images_with_depth))
        self.images_with_depth = self.images_with_depth[indices]
        self.labels = self.labels[indices]

learning_rate = 0.001
epochs = 50
batch_size = 32

image_with_depth_path=Path('train_image_with_depth_tensor.pt')
label_path=Path('train_label_tensor.pt')
val_image_with_depth_path=Path('val_image_with_depth_tensor.pt')
val_label_path=Path('val_label_tensor.pt')

# Create dataset instance
dataset = NYUDataset(image_with_depth_path, label_path)
val_dataset = NYUDataset(val_image_with_depth_path, val_label_path)

# Create data loader for training
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




# Define loss function (weighted cross-entropy is an option for imbalanced classes)
loss_fn = nn.CrossEntropyLoss()  # Adjust for weighted cross-entropy if needed

# Create device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

# Initialize model and optimizer
model = SegNet(in_channels=4, num_classes=37)  # Adjust out_channels=num_classes based on your dataset
initialize_weights(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset.labels = dataset.labels.to(torch.long)
val_dataset.labels = val_dataset.labels.to(torch.long)

def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, early_stopping_patience=5,resume_checkpoint=None):
    """
    Training loop for SegNet with NYU Depth V2 dataset
    """
    model.to(device)  # Move model to device (CPU or GPU)

    epoch_times = []
    best_val_loss = float('inf')
    no_improvement_count = 0

    # Create ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if resuming from a checkpoint
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resuming training from epoch {start_epoch}")

    else:
        start_epoch = 0

    for epoch in range(epochs):
        start_time = time.time()
        print(f"Epoch: {epoch+1}/{epochs}")

        # Shuffle training data loader after each epoch
        train_loader.dataset.shuffle_data()  # Assuming you have implemented a method to shuffle data in your dataset class

        model.train()  # Set model to training mode

        total_loss = 0.0
        for images_with_depth, labels in train_loader:  # Change here: Load images_with_depth instead of images
            images_with_depth, labels = images_with_depth.to(device), labels.to(device)  # Move data to device

            # Forward pass
            outputs = model(images_with_depth)  # Change here: Pass images_with_depth to model instead of images
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate loss

        # Calculate average loss per epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss:.4f}")
       
        #Validation loop here to monitor performance on unseen data

        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for val_images_with_depth, val_labels in val_loader:
                val_images_with_depth, val_labels = val_images_with_depth.to(device), val_labels.to(device)

                val_outputs = model(val_images_with_depth)
                val_loss += loss_fn(val_outputs, val_labels).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Perform early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_count = 0
            # Save the best model
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            torch.save(model, 'model.pth')
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                print(f"No improvement in validation loss for {early_stopping_patience} epochs. Stopping training.")
                break
        
        # Save checkpoint every epoch
        checkpoint_path = checkpoint_dir / f'epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, checkpoint_path)

        end_time = time.time()  # Record end time of epoch
        epoch_time = end_time - start_time  # Calculate time taken for epoch
        epoch_times.append(epoch_time)
        print(f"Time taken for Epoch {epoch+1}: {epoch_time:.2f} seconds")

    # Save the final model
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')
    print("Training finished. Model saved.")


    # Calculate average epoch time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"Average Training Duration: {avg_epoch_time:.2f} seconds") 

        


# Train the model
train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs)
