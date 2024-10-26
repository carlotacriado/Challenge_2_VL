import wandb
import torch

def test(model, dataloader, criterion, device, size = 256):
    model.eval()
    total_loss = 0

    for image, label, pat_id in dataloader:
        image = image.to(device)
        label = ((label + 1) / 2).to(device).view(-1, 1)

        #pat_id = pat_id.to(device)

        with torch.no_grad():
            outputs = model(image)

        test_loss = criterion(outputs, label)
        total_loss += test_loss.item()

    avg_loss = total_loss / len(dataloader)
    #print("Test loss = {:.6f}".format(avg_loss))

    return avg_loss