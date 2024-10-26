import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import wandb

# Custom libraries import
from TrainCNN import *
from TestCNN import *
from PatchClassificationCNN import *
from DataLoader import *

# MODEL TRAINING
# Parameters set-up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#wandb.login()

model = CNN_FC_Classifier().to(device)
# create an optimizer object
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
# mean-squared error loss
criterion = nn.BCELoss()
# learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

#early_stopping = EarlyStopping(patience=20, verbose=True)

data = Load_Data("HP_WSI-CoordAnnotatedPatches.xlsx", "Annotated")
dataset = MedicalImageDataset(data)

# Create a DataLoader for batching and shuffling
dataloader = DataLoader(dataset, batch_size=200, shuffle=True)

# ||TRAINING START||

#wandb.init(project="AlphaModel")

epochs = 100

for epoch in range(epochs):
    #wandb.watch(model, criterion, log="all", log_freq=10)

    print(f"Epoch {epoch+1}/{epochs}")
    loss_train = train(model, dataloader, optimizer, criterion, device, size = 256)
    print("Train loss = {:.6f}".format(loss_train))

    loss_val = test(model, dataloader, criterion, device, size = 256)
    print("Validation loss = {:.6f}".format(loss_val))

    #wandb.log({"train_loss": loss_train})

    scheduler.step(loss_val)

    # Check early stopping
    #early_stopping(loss_val, model)
    
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     torch.save(model.state_dict(), "AlphaModelCheckpoint.pth")
        # break

    if (epoch % 5) == 0:
        print("Saving Checkpoint")
        torch.save(model.state_dict(), f"Model{epoch}.pth")


# Saving the model
torch.save(model.state_dict(), 'Model.pth')

