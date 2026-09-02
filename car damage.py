import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


csv_path = r"C:\Users\SARAN K\OneDrive\Desktop\streamlit\structured_damage_dataset_no_imagepath.csv"
df = pd.read_csv(csv_path)

train_df = df[df['split'].str.lower() == "training"]
val_df   = df[df['split'].str.lower() == "validation"]
val_df, test_df = train_test_split(val_df, test_size=0.3, random_state=42, stratify=val_df['severity'])

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class CarDamageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.damage_classes = sorted(self.dataframe['impact_zone'].unique())
        self.severity_classes = sorted(self.dataframe['severity'].unique())
        self.damage_to_idx = {cls: idx for idx, cls in enumerate(self.damage_classes)}
        self.severity_to_idx = {cls: idx for idx, cls in enumerate(self.severity_classes)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['car_image_path']
        damage_label = self.damage_to_idx[row['impact_zone']]
        severity_label = self.severity_to_idx[row['severity']]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, damage_label, severity_label

train_dataset = CarDamageDataset(train_df, transform=train_transform)
val_dataset   = CarDamageDataset(val_df, transform=val_test_transform)
test_dataset  = CarDamageDataset(test_df, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ResNetDualHead(nn.Module):
    def __init__(self, num_damage_types, num_severity_levels):
        super(ResNetDualHead, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.damage_head = nn.Linear(num_features, num_damage_types)
        self.severity_head = nn.Linear(num_features, num_severity_levels)

    def forward(self, x):
        features = self.backbone(x)
        return self.damage_head(features), self.severity_head(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetDualHead(len(train_dataset.damage_classes), len(train_dataset.severity_classes)).to(device)

damage_counts = train_df['impact_zone'].value_counts()
damage_classes = sorted(train_df['impact_zone'].unique())
class_weights = [1.0 / damage_counts[cls] for cls in damage_classes]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion_damage = nn.CrossEntropyLoss(weight=class_weights)
criterion_severity = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

alpha, beta = 0.85, 0.15

epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss, total, correct_damage, correct_severity = 0, 0, 0, 0
    for images, damage_labels, severity_labels in train_loader:
        images, damage_labels, severity_labels = images.to(device), damage_labels.to(device), severity_labels.to(device)
        optimizer.zero_grad()
        damage_out, severity_out = model(images)
        loss_damage = criterion_damage(damage_out, damage_labels)
        loss_severity = criterion_severity(severity_out, severity_labels)
        loss = alpha * loss_damage + beta * loss_severity
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred_damage = damage_out.max(1)
        _, pred_severity = severity_out.max(1)
        total += images.size(0)
        correct_damage += pred_damage.eq(damage_labels).sum().item()
        correct_severity += pred_severity.eq(severity_labels).sum().item()
    train_acc_damage = 100. * correct_damage / total
    train_acc_severity = 100. * correct_severity / total

    
    model.eval()
    val_loss, val_total, val_correct_damage, val_correct_severity = 0, 0, 0, 0
    with torch.no_grad():
        for images, damage_labels, severity_labels in val_loader:
            images, damage_labels, severity_labels = images.to(device), damage_labels.to(device), severity_labels.to(device)
            damage_out, severity_out = model(images)
            loss_damage = criterion_damage(damage_out, damage_labels)
            loss_severity = criterion_severity(severity_out, severity_labels)
            loss = alpha * loss_damage + beta * loss_severity
            val_loss += loss.item()
            _, pred_damage = damage_out.max(1)
            _, pred_severity = severity_out.max(1)
            val_total += images.size(0)
            val_correct_damage += pred_damage.eq(damage_labels).sum().item()
            val_correct_severity += pred_severity.eq(severity_labels).sum().item()
    val_acc_damage = 100. * val_correct_damage / val_total
    val_acc_severity = 100. * val_correct_severity / val_total
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Damage Acc: {train_acc_damage:.2f}%, Severity Acc: {train_acc_severity:.2f}% "
          f"Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Val Damage Acc: {val_acc_damage:.2f}%, Val Severity Acc: {val_acc_severity:.2f}%")

torch.save(model.state_dict(), "car_damage_dualhead_resnet50_weighted.pth")
print("✅ Model saved as car_damage_dualhead_resnet50_weighted.pth")

all_damage_preds, all_damage_labels = [], []
all_severity_preds, all_severity_labels = [], []

with torch.no_grad():
    for images, damage_labels, severity_labels in test_loader:
        images, damage_labels, severity_labels = images.to(device), damage_labels.to(device), severity_labels.to(device)
        damage_out, severity_out = model(images)
        damage_preds = damage_out.argmax(1)
        severity_preds = severity_out.argmax(1)
        all_damage_preds.extend(damage_preds.cpu().numpy())
        all_damage_labels.extend(damage_labels.cpu().numpy())
        all_severity_preds.extend(severity_preds.cpu().numpy())
        all_severity_labels.extend(severity_labels.cpu().numpy())

print("=== Damage Type Classification ===")
print(classification_report(all_damage_labels, all_damage_preds, target_names=test_dataset.damage_classes))
cm_damage = confusion_matrix(all_damage_labels, all_damage_preds)
sns.heatmap(cm_damage, annot=True, fmt="d", xticklabels=test_dataset.damage_classes, yticklabels=test_dataset.damage_classes, cmap="Blues")
plt.title("Confusion Matrix - Damage Type")
plt.show()

print("=== Severity Classification ===")
print(classification_report(all_severity_labels, all_severity_preds, target_names=test_dataset.severity_classes))
cm_severity = confusion_matrix(all_severity_labels, all_severity_preds)
sns.heatmap(cm_severity, annot=True, fmt="d", xticklabels=test_dataset.severity_classes, yticklabels=test_dataset.severity_classes, cmap="Greens")
plt.title("Confusion Matrix - Severity")
plt.show()

infer_transform = val_test_transform

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = infer_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        damage_out, severity_out = model(image)
        damage_pred = damage_out.argmax(1).item()
        severity_pred = severity_out.argmax(1).item()
    return test_dataset.damage_classes[damage_pred], test_dataset.severity_classes[severity_pred]

img_path = r"C:\Users\SARAN K\OneDrive\Desktop\streamlit\image\training\03-severe\0001.JPEG"
damage, severity = predict(img_path)
print(f"Predicted Damage Type: {damage}, Severity: {severity}")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        self.model.eval()
        output_damage, _ = self.model(input_image)
        loss = output_damage[0, target_class]
        self.model.zero_grad()
        loss.backward()
        weights = self.gradients.mean(dim=[0,2,3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        return cam

image = Image.open(img_path).convert("RGB")
input_tensor = infer_transform(image).unsqueeze(0).to(device)

target_layer = model.backbone.layer4[-1]  
gradcam = GradCAM(model, target_layer)
cam = gradcam.generate_cam(input_tensor, target_class=0)  

plt.imshow(np.array(image))
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.title("Grad-CAM Visualization")
plt.show()
