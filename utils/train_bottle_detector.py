"""
Bottle Detection Training Pipeline
Uses Faster R-CNN with ResNet50 backbone from PyTorch (Apache 2.0 License)
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BottleDataset(Dataset):
    """
    Dataset format: 
    - images/: folder with images
    - annotations.json: COCO format or custom format
      {
        "images": [{"file_name": "img1.jpg", "id": 0, "height": 480, "width": 640}],
        "annotations": [{"image_id": 0, "bbox": [x, y, w, h], "category_id": 1}],
        "categories": [{"id": 1, "name": "bottle"}]
      }
    """

    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get annotations for this image
        img_id = img_info['id']
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:
            # COCO format: [x, y, width, height] -> convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            # Albumentations format
            transformed = self.transforms(
                image=img,
                bboxes=boxes.numpy(),
                labels=labels.numpy()
            )
            img = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
        else:
            img = T.ToTensor()(img)

        return img, target


def get_transform(train=True):
    """Data augmentation pipeline"""
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.Resize(800, 800),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(800, 800),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_model(num_classes=3):
    """
    Create Faster R-CNN model
    num_classes = 1 (background) + 1 (bottle) + 1 (cap) = 3
    """
    # Load pretrained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch}], Step [{i+1}/{len(data_loader)}], Loss: {losses.item():.4f}")

    return total_loss / len(data_loader)


def validate(model, data_loader, device):
    total_loss = 0.0

    model.train()   # ❗ ВАЖНО
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    model.eval()    # връщаме обратно
    return total_loss / len(data_loader)


def main():
    # Configuration
    config = {
        'data_dir': 'data/images',
        'train_annotations': 'data/train_annotations.json',
        'val_annotations': 'data/val_annotations.json',
        'num_classes': 3,  # background + bottle + cap
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'num_workers': 4,
        'save_dir': 'checkpoints'
    }

    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = BottleDataset(
        config['data_dir'],
        config['train_annotations'],
        transforms=get_transform(train=True)
    )

    val_dataset = BottleDataset(
        config['data_dir'],
        config['val_annotations'],
        transforms=get_transform(train=False)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    # Create model
    model = get_model(num_classes=config['num_classes'])
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*50}")

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        print(f"Average Training Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Update learning rate
        lr_scheduler.step()

        # Save checkpoint
        # checkpoint_path = os.path.join(config['save_dir'], f'model_epoch_{epoch+1}.pth')
        # torch.save({
        #     'epoch': epoch + 1,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'train_loss': train_loss,
        #     'val_loss': val_loss,
        # }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Saved best model with validation loss: {val_loss:.4f}")

    print("\nTraining completed!")
    print(f"Best model saved to: {best_model_path}")


if __name__ == '__main__':
    main()
