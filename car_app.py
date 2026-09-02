import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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
damage_classes = ["Front","Front-Left","Front-Right","Rear","Roof","Side-Left","Side-Right","Underbody"]
severity_classes = ["Minor","Moderate","Severe"]

model = ResNetDualHead(len(damage_classes), len(severity_classes)).to(device)
model.load_state_dict(torch.load("car_damage_dualhead_resnet50_weighted.pth", map_location=device))
model.eval()


infer_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


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
        self.target_layer.register_full_backward_hook(backward_hook)

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

st.title("🚗 Vehicle Damage Detection App")
st.write("Upload a vehicle photo to predict damage type and severity with Grad-CAM visualization.")

uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = infer_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        damage_out, severity_out = model(input_tensor)
        damage_probs = torch.softmax(damage_out, dim=1)[0]
        severity_probs = torch.softmax(severity_out, dim=1)[0]

        damage_pred = damage_out.argmax(1).item()
        severity_pred = severity_out.argmax(1).item()

    st.subheader("Prediction Results")
    st.write(f"**Damage Type:** {damage_classes[damage_pred]} ({damage_probs[damage_pred]*100:.2f}%)")
    st.write(f"**Severity:** {severity_classes[severity_pred]} ({severity_probs[severity_pred]*100:.2f}%)")

    st.info(f"Detected: {damage_classes[damage_pred]} – {severity_classes[severity_pred]} damage → Suggested next step: Route to body-shop estimate.")

    target_layer = model.backbone.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor, target_class=damage_pred)

    fig, ax = plt.subplots()
    ax.imshow(np.array(image))
    ax.imshow(cam, cmap='jet', alpha=0.5)
    ax.set_title("Grad-CAM Heatmap")
    st.pyplot(fig)
