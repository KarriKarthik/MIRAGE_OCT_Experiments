import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler # for mixed precision
import torch.optim as optim

# The optimizer only receives the parameters of the head
optimizer = optim.SGD(model.head.parameters(),lr=0.5, momentum=0.5)

# Categorical cross-entropy loss. Assumes softmax classifier. Seperate defination not needed
criterion = nn.CrossEntropyLoss()

# Data Transformations
data_transforms = transforms.Compose([
    transforms.Resize((512, 512)), # 512 is dimension of pre-trained foundation model
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

# Data Loading
data_dir = '/content/train_data/content/drive/MyDrive/Kaggle_2/oct_2017/train/'
image_dataset = datasets.ImageFolder(data_dir, data_transforms) # Finds image classes from subdirectories

# Data loader with multiple workers to save time and use of GPU for data transfer 
dataloader = torch.utils.data.DataLoader(
    image_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


# Training Loop 
num_epochs = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
scaler = torch.GradScaler("cuda") # Regrading mixed precision training (float16)

print(f"Starting training on {device} with mixed precision.")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        
        with torch.autocast(device_type="cuda"): # torch.float16 is default
            x = model.model.input_adapters['bscan'](inputs)
            x = model.model.encoder(x)
            x_pooled = torch.mean(x, dim=1)
            outputs = model.head(x_pooled)
            loss = criterion(outputs, labels)

            # Calculate predictions for accuracy during training
            _, preds = torch.max(outputs, 1)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)


    epoch_loss = running_loss / len(image_dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")


print("====== Training Completed ====")
