import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models
from tqdm import tqdm

DATA_DIR = Path("data")
OUT_PATH = Path("weights/deepfake_efficientnet_b0.pth")
IMG_SIZE = 224
BATCH = 16
EPOCHS = 3
LR = 3e-4

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # folder structure for ImageFolder must be:
    # data/real/*.jpg, data/fake/*.jpg
    # but ImageFolder expects class folders under a root directory.
    # We'll create a temporary root: data_train/{real,fake}
    root = DATA_DIR
    ds = datasets.ImageFolder(root=str(root), transform=tfm)

    # Ensure class_to_idx maps: {'fake':1,'real':0} or similar
    print("Classes:", ds.classes, ds.class_to_idx)

    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(EPOCHS):
        pbar = tqdm(loader, desc=f"epoch {ep+1}/{EPOCHS}")
        total = 0.0
        for x, y in pbar:
            x = x.to(device)
            y = y.float().to(device)  # 0/1
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=total / max(1, len(pbar)))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_PATH)
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
