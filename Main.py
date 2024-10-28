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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# mean-squared error loss
criterion = nn.BCELoss()
# learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

#early_stopping = EarlyStopping(patience=20, verbose=True)

excel = "/export/fhome/vlia/HelicoDataSet/HP_WSI-CoordAnnotatedPatches.xlsx"
data = "/export/fhome/vlia/HelicoDataSet/CrossValidation/Annotated/"
data = Load_Data(excel, data)
dataset = MedicalImageDataset(data)


# Split the dataset into train, validation and test sets
size = int(0.8 * len(dataset))
test_size = len(dataset) - size
train_size = int(0.8 * size)
validation_size = size - train_size

train_dataset, test_dataset, validation_dataset = random_split(dataset, [train_size, test_size, validation_size])

# Create DataLoaders for train validation and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# ||TRAINING START||

#wandb.init(project="AlphaModel")

epochs = 1000

for epoch in range(epochs):
    #wandb.watch(model, criterion, log="all", log_freq=10)

    print(f"Epoch {epoch+1}/{epochs}")
    loss_train = train(model, train_loader, optimizer, criterion, device, size = 256)
    print("Train loss = {:.6f}".format(loss_train))

    loss_val = test(model, validation_loader, criterion, device, size = 256)
    print("Validation loss = {:.6f}".format(loss_val))

    #wandb.log({"train_loss": loss_train})

    scheduler.step(loss_val)

    # Check early stopping
    #early_stopping(loss_val, model)
    
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     torch.save(model.state_dict(), "AlphaModelCheckpoint.pth")
        # break

    '''if (epoch % 5) == 0:
        print("Saving Checkpoint")
        torch.save(model.state_dict(), f"Model{epoch}.pth")'''


# Saving the model
torch.save(model.state_dict(), 'Model.pth')
print("Training finished!")
