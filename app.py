import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import load_models, predict_caption, predict_segmentation

# Load models and vocab
caption_model, seg_model, vocab = load_models()

st.title("Image Captioning + Segmentation")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)

    # Predictions
    caption = predict_caption(image_tensor, caption_model, vocab)
    masks = predict_segmentation(image_tensor, seg_model)

    # Display
    st.markdown(f"**Predicted Caption:** {caption}")

    fig, ax = plt.subplots()
    ax.imshow(image_tensor.permute(1, 2, 0).numpy())
    for mask in masks:
        ax.imshow(mask[0].cpu(), alpha=0.3, cmap='Reds')
    st.pyplot(fig)
