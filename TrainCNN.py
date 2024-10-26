import wandb
import torch

def train(model, dataloader, optimizer, criterion, device, size = 256):
    model.train() # Set model to train mode
    total_loss = 0 # Initialize total loss
    for image, label, pat_id in dataloader:
        image = image.to(device)
        label = ((label + 1) / 2).to(device).view(-1, 1)
        #pat_id = pat_id.to(device)

        optimizer.zero_grad()

        outputs = model(image)

        # Calculate loss and gradient
        train_loss = criterion(outputs, label)
        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item()

    avg_loss = total_loss / len(dataloader)
    #print("Train loss = {:.6f}".format(avg_loss))

    return avg_loss