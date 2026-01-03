import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import UNet
from torchvision import transforms

# 1. Page Config
st.set_page_config(page_title="Flood Detector", page_icon="🌊")
st.title("🛰️ Satellite Flood Detection AI")
st.write("Upload a satellite image to detect water bodies instantly.")

# 2. Load Model (Cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    model = UNet(n_channels=3, n_classes=1)
    # Map_location ensures it works on CPU even if trained on GPU
    model.load_state_dict(torch.load("flood_model_final.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Image Preprocessing Logic
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# 4. The App Interface
uploaded_file = st.file_uploader("Choose a Satellite Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Original
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create columns for side-by-side view
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Detect Flood'):
        with st.spinner('Analyzing satellite data...'):
            # Predict
            img_tensor = process_image(image)
            with torch.no_grad():
                output = model(img_tensor)
                # Sigmoid gives probability 0.0 to 1.0
                # We turn anything > 0.5 into 1.0, else 0.0
                prediction = (torch.sigmoid(output) > 0.5).float()
            
            # Convert back to image
            pred_mask = prediction.squeeze().cpu().numpy()
            
            # CRITICAL FIX: Multiply by 255 to make water WHITE
            pred_mask = (pred_mask * 255).astype(np.uint8)
            
            # Show Result in the second column
            with col2:
                st.image(pred_mask, caption='Predicted Flood Mask', clamp=True, channels='GRAY', use_container_width=True)
            
            st.success("Analysis Complete!")
            st.write(f"**White areas** represent detected water bodies.")