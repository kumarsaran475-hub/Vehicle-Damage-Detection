import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import resnet50

st.set_page_config(page_title="Vehicle Damage AI", page_icon="🚗", layout="wide")

MODEL_PATH = "car_damage_dualhead_resnet50_weighted.pth"
DAMAGE_CLASSES = ["Front", "Front-Left", "Front-Right", "Rear",
                  "Roof", "Side-Left", "Side-Right", "Underbody"]
SEVERITY_CLASSES = ["Minor", "Moderate", "Severe"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DISPLAY_SIZE = 1024  
DEFAULT_SEVERITY_COST = {"Minor": 11500, "Moderate": 92500, "Severe": 335000}
AREA_MULTIPLIER = {z: 1.0 for z in ["Front", "Front-Left", "Front-Right", "Rear",
                                    "Roof", "Side-Left", "Side-Right", "Underbody"]}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ResNetDualHead(nn.Module):
    def __init__(self, num_damage_types, num_severity_levels):
        super().__init__()
        self.backbone = resnet50(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.damage_head = nn.Linear(num_features, num_damage_types)
        self.severity_head = nn.Linear(num_features, num_severity_levels)

    def forward(self, x):
        features = self.backbone(x)
        return self.damage_head(features), self.severity_head(features)

@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str) -> ResNetDualHead:
    """Loaded once per server process instead of on every Streamlit rerun."""
    model = ResNetDualHead(len(DAMAGE_CLASSES), len(SEVERITY_CLASSES))
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()

def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB") 
    return img

def norm_entropy(p: np.ndarray) -> float:
    """0 = fully certain, 1 = completely uncertain."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

@st.cache_data(show_spinner=False)
def run_inference(_model: ResNetDualHead, data: bytes) -> dict:
    image = load_image(data)
    x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        d_out, s_out = _model(x)
    d_probs = torch.softmax(d_out, 1)[0].cpu().numpy()
    s_probs = torch.softmax(s_out, 1)[0].cpu().numpy()
    return {"damage_probs": d_probs, "severity_probs": s_probs}

def estimate_cost(damage: str, severity_probs: np.ndarray, sev_cost: dict) -> float:
    """Expected cost = sum_i P(severity_i) * cost_i, scaled by damaged area."""
    base = sum(p * sev_cost[c] for p, c in zip(severity_probs, SEVERITY_CLASSES))
    return float(base * AREA_MULTIPLIER[damage])

def recommend_action(severity: str, conf: float, threshold: float) -> str:
    if conf < threshold:
        return "⚠️ Low confidence - manual inspection"
    return {
        "Minor": "✅ Fast-track minor repair",
        "Moderate": "🔧 Route to body-shop estimate",
        "Severe": "🚨 Senior adjuster + full inspection",
    }[severity]

def compute_cam(model, x, head: str, class_idx: int, method: str) -> np.ndarray:
    """
    Hooks are attached and removed on every call, so they never pile up across
    Streamlit reruns. A tensor hook is used for gradients (safe with in-place ReLU).
    """
    layer = model.backbone.layer4[-1]
    store = {}

    def fwd_hook(_, __, out):
        store["acts"] = out
        out.register_hook(lambda g: store.__setitem__("grads", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            d_out, s_out = model(x)
            logits = d_out if head == "Damage area" else s_out
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    acts = store["acts"].detach()[0]   
    grads = store["grads"].detach()[0]  

    if method == "Grad-CAM":
        weights = grads.mean(dim=(1, 2))
    else: 
        g2, g3 = grads ** 2, grads ** 3
        denom = 2 * g2 + acts.sum(dim=(1, 2), keepdim=True) * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))

    cam = F.relu((weights[:, None, None] * acts).sum(0))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)  
    return cam.cpu().numpy()

def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float, cmap: str):
    """Upsample CAM to image size and blend it, weighted by activation strength."""
    cam_t = torch.from_numpy(cam)[None, None]
    cam_up = F.interpolate(cam_t, size=(image.height, image.width),
                           mode="bilinear", align_corners=False)[0, 0].numpy()
    cam_up = np.clip(cam_up, 0, 1)
    heat = (colormaps[cmap](cam_up)[..., :3] * 255).astype(np.float32)
    base = np.asarray(image, dtype=np.float32)
    w = (alpha * cam_up)[..., None] 
    out = base * (1 - w) + heat * w
    return Image.fromarray(out.astype(np.uint8)), cam_up

def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Min. confidence for auto-decision", 0.30, 0.99, 0.60, 0.01)

    st.subheader("Grad-CAM")
    cam_method = st.radio("Method", ["Grad-CAM", "Grad-CAM++"], horizontal=True)
    cam_alpha = st.slider("Overlay strength", 0.1, 1.0, 0.6, 0.05)
    cam_cmap = st.selectbox("Colormap", ["jet", "inferno", "magma", "viridis", "turbo"])

    with st.expander("💰 Cost model (illustrative)"):
        currency = st.text_input("Currency symbol", "₹")
        sev_cost = {c: st.number_input(f"{c} base cost", min_value=0,
                                       value=DEFAULT_SEVERITY_COST[c], step=1000)
                    for c in SEVERITY_CLASSES}

    st.caption(f"Device: `{DEVICE}`")

st.title("🚗 Vehicle Damage Detection")
st.write("Upload one or more vehicle photos to get the damage area, severity, "
         "an uncertainty score, a cost estimate and an explainable heatmap.")

try:
    model = load_model(MODEL_PATH)
except Exception as e:  
    st.error(f"Could not load model from `{MODEL_PATH}`: {e}")
    st.stop()

files = st.file_uploader("Upload vehicle images", type=["jpg", "jpeg", "png"],
                         accept_multiple_files=True)
if not files:
    st.info("👆 Upload at least one image to begin.")
    st.stop()

rows, results = [], []
progress = st.progress(0.0, text="Analysing images...")
for i, f in enumerate(files):
    data = f.getvalue()
    res = run_inference(model, data)
    d_idx, s_idx = int(res["damage_probs"].argmax()), int(res["severity_probs"].argmax())
    d_name, s_name = DAMAGE_CLASSES[d_idx], SEVERITY_CLASSES[s_idx]
    d_conf, s_conf = float(res["damage_probs"][d_idx]), float(res["severity_probs"][s_idx])
    unc = (norm_entropy(res["damage_probs"]) + norm_entropy(res["severity_probs"])) / 2
    cost = estimate_cost(d_name, res["severity_probs"], sev_cost)

    results.append({"name": f.name, "data": data, **res,
                    "d_idx": d_idx, "s_idx": s_idx})
    rows.append({
        "File": f.name,
        "Damage area": d_name,
        "Area conf. %": d_conf * 100,
        "Severity": s_name,
        "Severity conf. %": s_conf * 100,
        "Uncertainty": round(unc, 3),
        f"Est. cost ({currency})": round(cost),
        "Suggested action": recommend_action(s_name, min(d_conf, s_conf), conf_threshold),
    })
    progress.progress((i + 1) / len(files), text=f"Analysing {f.name}")
progress.empty()

df = pd.DataFrame(rows)

tab_overview, tab_detail = st.tabs(["📊 Batch overview", "🔍 Detailed analysis"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", len(df))
    c2.metric("Severe cases", int((df["Severity"] == "Severe").sum()))
    c3.metric("Needs manual review",
              int(df["Suggested action"].str.contains("Low confidence").sum()))
    c4.metric(f"Total est. cost ({currency})", f"{df[f'Est. cost ({currency})'].sum():,.0f}")

    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Area conf. %": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100),
            "Severity conf. %": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100),
        },
    )
    st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False).encode(),
                       "damage_report.csv", "text/csv")

    if len(df) > 1:
        a, b = st.columns(2)
        a.caption("Damage area distribution")
        a.bar_chart(df["Damage area"].value_counts())
        b.caption("Severity distribution")
        b.bar_chart(df["Severity"].value_counts())

with tab_detail:
    labels = [f"{i + 1}. {r['name']}" for i, r in enumerate(results)]
    pick = st.selectbox("Select image", range(len(results)), format_func=lambda i: labels[i])
    r = results[pick]

    image = load_image(r["data"])
    image.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE))

    h1, h2 = st.columns(2)
    head = h1.radio("Explain which head?", ["Damage area", "Severity"], horizontal=True)
    classes = DAMAGE_CLASSES if head == "Damage area" else SEVERITY_CLASSES
    pred_idx = r["d_idx"] if head == "Damage area" else r["s_idx"]
    target_label = h2.selectbox("Target class", classes, index=pred_idx,
                                help="Default is the predicted class. Pick another class "
                                     "to see what the model would look at for it.")
    target_idx = classes.index(target_label)

    x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    cam = compute_cam(model, x, head, target_idx, cam_method)
    overlay, cam_up = overlay_cam(image, cam, cam_alpha, cam_cmap)

    left, right = st.columns(2)
    left.image(image, caption="Original", use_container_width=True)
    right.image(overlay, caption=f"{cam_method} → {head}: {target_label}",
                use_container_width=True)

    ys, xs = np.unravel_index(cam_up.argmax(), cam_up.shape)
    st.caption(f"Peak attention at ≈ ({xs / cam_up.shape[1]:.0%} x, "
               f"{ys / cam_up.shape[0]:.0%} y) of the image. "
               f"{(cam_up > 0.5).mean():.0%} of the image is above 50% activation.")

    st.download_button("⬇️ Download heatmap (PNG)", to_png(overlay),
                       f"gradcam_{r['name'].rsplit('.', 1)[0]}.png", "image/png")

    st.subheader("Class probabilities")
    p1, p2 = st.columns(2)
    p1.caption("Damage area")
    p1.bar_chart(pd.Series(r["damage_probs"], index=DAMAGE_CLASSES))
    p2.caption("Severity")
    p2.bar_chart(pd.Series(r["severity_probs"], index=SEVERITY_CLASSES))

    row = df.iloc[pick]
    conf = min(row["Area conf. %"], row["Severity conf. %"]) / 100
    msg = (f"**{row['Damage area']} – {row['Severity']}** damage  |  "
           f"Est. cost: {currency}{row[f'Est. cost ({currency})']:,.0f}  |  "
           f"{row['Suggested action']}")
    (st.success if conf >= conf_threshold else st.warning)(msg)

st.caption("⚠️ Estimates are model outputs for decision support only - "
           "always confirm with a qualified inspector.")