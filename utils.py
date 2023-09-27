import torch
import torchvision
from torchmetrics import Accuracy
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"]) # loads the parameter values from the checkpoint state dict into the model

def get_loaders(
                train_dir,
                train_maskdir,
                val_dir,
                val_maskdir,
                batch_size,
                train_transform,
                val_transform,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    model.eval() #eval mode disables dropout & batch norm

    with torch.no_grad(): #without gradient tracking
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float() #converts the probability values into binary predictions
            Accuracy(preds, y)

    accuracy = Accuracy.compute().item()
    print(f"Accuracy: {accuracy:.2f}%")
    model.train()
    
    return accuracy

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png") # Save a given Tensor into an image file.
        
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png") #saves the true label

    model.train()