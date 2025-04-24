import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import random
import time

# Set device 
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Define directories
base_directory = os.path.expanduser("~/Documents/OEIS_Sequence_Repository")
linear_directory = os.path.join(base_directory, "Linear_Scatterplots")
log_directory = os.path.join(base_directory, "Logarithmic_Scatterplots")

# Custom Dataset for OEIS images
class OEISDataset(Dataset):
    def __init__(self, linear_dir, log_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []  # 0 for linear, 1 for logarithmic
        self.sequence_ids = []
        
        # Load linear images (label 0)
        for file in os.listdir(linear_dir):
            if file.endswith(".png"):
                file_path = os.path.join(linear_dir, file)
                sequence_id = os.path.splitext(file)[0]
                self.images.append(file_path)
                self.labels.append(0)
                self.sequence_ids.append(sequence_id)
        
        # Load logarithmic images (label 1)
        for file in os.listdir(log_dir):
            if file.endswith(".png"):
                file_path = os.path.join(log_dir, file)
                sequence_id = os.path.splitext(file)[0]
                self.images.append(file_path)
                self.labels.append(1)
                self.sequence_ids.append(sequence_id)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        sequence_id = self.sequence_ids[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, sequence_id

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Create dataset
dataset = OEISDataset(linear_directory, log_directory, transform=transform)

# Split dataset into train, validation, and test sets (70/15/15 split)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

# Create data loaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: Linear and Logarithmic
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128x128 -> 64x64
        x = self.pool(F.relu(self.conv2(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv3(x)))  # 32x32 -> 16x16
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model
model = CNN().to(device)
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to display sample images
def display_sample_images(dataset, num_images=50):
    # Create a random selection of images
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    
    # Get the selected images
    selected_images = []
    selected_ids = []
    selected_labels = []
    
    for idx in indices:
        img, label, seq_id = dataset[idx]
        selected_images.append(img)
        selected_ids.append(seq_id)
        selected_labels.append(label)
    
    # Display images in a grid
    rows, cols = 10, 5  # 10x5 layout
    fig, axes = plt.subplots(rows, cols, figsize=(15, 30))
    
    for i, (img, seq_id, label) in enumerate(zip(selected_images, selected_ids, selected_labels)):
        if i < rows * cols:  # Make sure we don't exceed the grid size
            axes.flatten()[i].imshow(img.squeeze().numpy(), cmap="gray")
            label_text = "Linear" if label == 0 else "Logarithmic"
            axes.flatten()[i].set_title(f"{label_text}\n{seq_id}", fontsize=8)
            axes.flatten()[i].axis("off")
    
    # Hide unused subplots
    for i in range(len(selected_images), rows * cols):
        axes.flatten()[i].axis("off")
    
    plt.tight_layout()
    plt.show()

# Display sample images
display_sample_images(dataset)

# Training loop
num_epochs = 10
train_losses = []
val_accuracies = []

# Add timer for training
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {avg_train_loss:.4f} - Validation accuracy: {val_accuracy:.2f}%')

# Calculate training time
training_time = time.time() - start_time
hours, remainder = divmod(training_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f'Finished training in {int(hours)}h {int(minutes)}m {seconds:.2f}s')

# Evaluate on test set
model.eval()
test_correct = 0
test_total = 0

# Create a list to store predictions and sequence IDs
predictions = []
sequence_ids = []
true_labels = []

with torch.no_grad():
    for inputs, labels, seq_ids in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # Store predictions and sequence IDs
        predictions.extend(predicted.cpu().numpy())
        sequence_ids.extend(seq_ids)
        true_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * test_correct / test_total
print(f'Test accuracy: {test_accuracy:.2f}%')

# Print detailed results for each test image
print("\nDetailed Test Results:")
print("Sequence ID | True Class | Predicted Class | Correct?")
print("-" * 60)
for i, (seq_id, true_label, pred_label) in enumerate(zip(sequence_ids, true_labels, predictions)):
    true_class = "Linear" if true_label == 0 else "Logarithmic"
    pred_class = "Linear" if pred_label == 0 else "Logarithmic"
    correct = "✓" if true_label == pred_label else "✗"
    print(f"{seq_id} | {true_class} | {pred_class} | {correct}")

# Plot training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()