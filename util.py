from models import YoloModel
import torch
import torch.nn as nn
from args import args_parser
import os
from PIL import Image
import torch.optim as optim
from data_preprocess import generate_dataloader
from criterion import YOLOLoss
from tqdm import tqdm


args = args_parser()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

train_loader, test_loader = generate_dataloader()

model = YoloModel(num_classes=5)  # num_classes = number of classes in your dataset
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = YOLOLoss(num_classes=5)


# Training loop
for epoch in tqdm(range(args.epochs)):
    # Train
    model.train()
    train_loss = 0.0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        print('loss=', loss.item())
    train_loss /= len(train_loader)

torch.save(model.state_dict(), "yolov3.pth")
