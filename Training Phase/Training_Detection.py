import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from pytorch_metric_learning import losses, miners

dataset_path = "<Path To Your Dataset Here>"
model_save_path = "<Your Saving Path Here>.plt"

# Define transformations for your dataset, including data augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your dataset
dataset = datasets.ImageFolder(root=dataset_path)

# Split dataset into training and validation sets (80% train, 20% val)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Apply transformations to the train and validation datasets
train_dataset.dataset.transform = transform_train
val_dataset.dataset.transform = transform_val

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load the pretrained ResNet-18 model and modify it
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 128)  # Use a smaller embedding size for metric learning

# Define the loss function and mining function
loss_func = losses.TripletMarginLoss(margin=0.2)
miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.0001)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training and validation loop
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        embeddings = model(inputs)
        hard_pairs = miner(embeddings, labels)
        loss = loss_func(embeddings, labels, hard_pairs)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            embeddings = model(inputs)
            hard_pairs = miner(embeddings, labels)
            val_loss = loss_func(embeddings, labels, hard_pairs)

            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Save the model if validation loss has decreased
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f'Saved model with val loss: {avg_val_loss:.4f}')

print('Training complete')
