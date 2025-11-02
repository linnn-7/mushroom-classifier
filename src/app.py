import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import FineTuneResNet18 
import json 
import os 

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

# Load the Knowledge Base from JSON ---
@st.cache_data  # Cache the KB so it doesn't reload on every interaction
def load_knowledge_base():
    # 2. Get the absolute path to the directory app.py is in
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 3. Join that path with the filename
    json_path = os.path.join(script_dir, "knowledge_base.json")
    
    try:
        with open(json_path, "r") as f: # 4. Use this new, absolute path
            return json.load(f)
    except FileNotFoundError:
        st.error(f"FATAL: knowledge_base.json not found. Looked for it at: {json_path}")
        return {} 
    except json.JSONDecodeError:
        st.error("FATAL: knowledge_base.json is not a valid JSON. Please check its syntax.")
        return {}

mushroom_kb = load_knowledge_base()

# --- 3. Update predict_mushroom to get confidence and full KB object ---
def predict_mushroom(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        # Get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        
    class_name = classes[pred_idx.item()]
    info_object = mushroom_kb.get(class_name, None) # Get the full object
    
    return class_name, info_object, confidence.item()

# --- 4. Streamlit UI (Completely updated) ---
st.title("Mushroom Classifier Chatbot 🍄")
st.write("Upload an image of a mushroom for a prediction. **Warning: Never eat a mushroom based on this app's identification alone.**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None and mushroom_kb: # Only run if KB loaded
    image = Image.open(uploaded_file).convert("RGB")
    
    # New 2-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Placeholder while predicting
    with st.spinner("Classifying..."):
        pred_class, info, confidence = predict_mushroom(image)
    
    # Display results in the second column
    with col2:
        st.subheader(f"Initial Prediction: **{pred_class}**")
        st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")

        if info:
            # --- THIS IS YOUR "MEMBER B" REASONING LAYER ---
            
            # SPECIAL CASE: The Agaricus vs. Amanita problem
            if pred_class == "Agaricus":
                st.error(f"**CRITICAL SAFETY WARNING**")
                st.warning(f"The model predicts **{pred_class}**, but this mushroom is visually identical to the **deadly 'Destroying Angel' (Amanita)**.")
                st.write("Please answer these questions to confirm the ID. **Never consume a mushroom based on this app alone.**")
                
                st.markdown("---")

                # Ask the "Expert Rule" questions from your KB
                spore_color = st.selectbox(
                    "1. What color is the spore print? (Tap the cap on paper for an hour)",
                    ("I don't know", "White", "Dark Brown / Chocolate", "Pink", "Other")
                )
                
                volva_present = st.radio(
                    "2. Is there a sac-like cup (called a 'volva') at the very base of the stem, possibly under the soil?",
                    ("I don't know", "Yes, I see a sac-like cup", "No, there is no cup")
                )

                # Provide a reasoned output based on the rules
                if spore_color == "White" or volva_present == "Yes, I see a sac-like cup":
                    st.error("**DO NOT EAT. This strongly matches the deadly Amanita.**")
                    st.write("Amanitas have a **white** spore print and a **volva**. Agaricus do not.")
                elif spore_color == "Dark Brown / Chocolate" and volva_present == "No, there is no cup":
                    st.success("**This confirms the Agaricus ID.**")
                    st.write("Agaricus mushrooms are characterized by a **dark brown** spore print and **no volva**.")
                else:
                    st.info("Please complete the checks to get a safer recommendation.")
            
            # NORMAL CASE: For all other mushrooms
            else:
                # Display safety warning based on edibility
                if info['edibility'] in ["Deadly Poisonous", "Varies (Some Deadly)", "Varies (Some Toxic)"]:
                    st.error(f"**Safety Warning:** {info['safety_warning']}")
                elif info['edibility'] == "Varies (Many Choice Edibles)":
                    st.warning(f"**Safety Warning:** {info['safety_warning']}")
                else:
                    st.success(f"**Edibility:** {info['edibility']}")

                # Use expander for detailed info
                with st.expander("Full Mushroom Information"):
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown("**Distinguishing Features:**")
                    for feature in info['distinguishing_features']:
                        st.markdown(f"- {feature}")
                    st.markdown(f"**Taxonomy:** {info['taxonomy']['family']}")
        
        else:
            st.error(f"No information found for '{pred_class}' in the knowledge_base.json.")