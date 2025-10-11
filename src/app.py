import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import FineTuneResNet18 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("mushroom_model.pt", map_location=device)
classes = checkpoint['classes']

num_classes = len(classes)
model = FineTuneResNet18(num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mushroom_info = {
    "Agaricus": "Common edible mushroom. Often found in lawns and meadows.",
    "Amanita": "Some species are deadly poisonous. Avoid if unsure.",
    "Boletus": "Edible mushrooms with thick caps. Many are prized.",
    "Cortinarius": "Some species toxic. Do not eat wild mushrooms unless confident.",
    "Entoloma": "Some species are poisonous. Identification is critical.",
    "Hygrocybe": "Mostly small and brightly colored mushrooms, generally edible.",
    "Lactarius": "Edible mushrooms exude milky latex when cut.",
    "Russula": "Many are edible, but some are toxic. Handle with care.",
    "Suillus": "Mostly edible, often found under conifers."
}

def predict_mushroom(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, pred = output.max(1)
    class_name = classes[pred.item()]
    info = mushroom_info.get(class_name, "No info available.")
    return class_name, info

# --- Streamlit UI ---
st.title("Mushroom Classifier Chatbot")
st.write("Upload an image of a mushroom and get the predicted species and information.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Show uploaded image in center
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Placeholder while predicting
    with st.spinner("Classifying..."):
        pred_class, info = predict_mushroom(image)
    
    # Display results after prediction
    st.success(f"Predicted Class: **{pred_class}**")
    
    # Use expander for detailed info
    with st.expander("Mushroom Information"):
        st.write(info)

