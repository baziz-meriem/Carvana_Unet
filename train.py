import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 3
NUM_EPOCHS = 1
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):#iterates over the batches of images and their masks 
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop  to display the current loss value alongside the progress information
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose( #Composes several transforms together.
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),#p=1 --> apply on every image
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,#value to devide by to get values between 0 and 1
            ),
            ToTensorV2(),#convert images to tensors
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()#has sigmoid integrated +its more perfomant
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
                                    TRAIN_IMG_DIR,
                                    TRAIN_MASK_DIR,
                                    VAL_IMG_DIR,
                                    VAL_MASK_DIR,
                                    BATCH_SIZE,
                                    train_transform,
                                    val_transforms,
                                )

    if LOAD_MODEL:#load a saved pytorch model checkpoint
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        print("checking accuracy of the checkpoint==>")
        check_accuracy(val_loader, model, device=DEVICE)
    else:

        for epoch in range(NUM_EPOCHS):
            train_fn(train_loader, model, optimizer, loss_fn)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            # check accuracy
            check_accuracy(val_loader, model, device=DEVICE)

            # print some examples to a folder
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )


if __name__ == "__main__":
    main()