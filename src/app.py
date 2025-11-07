import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import FineTuneResNet18 
import json 
import os
import base64
from openai import OpenAI 

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "knowledge_base.json")
    
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"FATAL: knowledge_base.json not found. Looked for it at: {json_path}")
        return {} 
    except json.JSONDecodeError:
        st.error("FATAL: knowledge_base.json is not a valid JSON. Please check its syntax.")
        return {}

mushroom_kb = load_knowledge_base()

# Load Sporacle avatar path and image
@st.cache_data
def get_sporacle_avatar_path():
    """Get the path to the Sporacle avatar image"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    avatar_path = os.path.join(script_dir, "sporacle.gif")
    if os.path.exists(avatar_path):
        return avatar_path
    return None

@st.cache_data
def load_sporacle_image():
    """Load the Sporacle image for display"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    avatar_path = os.path.join(script_dir, "sporacle.gif")
    try:
        return Image.open(avatar_path)
    except FileNotFoundError:
        return None

@st.cache_data
def get_sporacle_gif_base64():
    """Get base64 encoded GIF for HTML display"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    avatar_path = os.path.join(script_dir, "sporacle.gif")
    try:
        with open(avatar_path, "rb") as gif_file:
            gif_bytes = gif_file.read()
            encoded_gif = base64.b64encode(gif_bytes).decode("utf-8")
            return encoded_gif
    except FileNotFoundError:
        return None

sporacle_avatar_path = get_sporacle_avatar_path()
sporacle_image = load_sporacle_image()
sporacle_gif_base64 = get_sporacle_gif_base64()

# Load mushroom parts diagram (if available)
@st.cache_data
def get_mushroom_diagram_path():
    """Get the path to the mushroom parts diagram"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    diagram_path = os.path.join(script_dir, "mushroom_diagram.png")
    if os.path.exists(diagram_path):
        return diagram_path
    # Also check for other common formats
    for ext in [".jpg", ".jpeg", ".gif", ".webp"]:
        alt_path = os.path.join(script_dir, f"mushroom_diagram{ext}")
        if os.path.exists(alt_path):
            return alt_path
    return None

mushroom_diagram_path = get_mushroom_diagram_path()

# --- Prediction Function ---
def predict_mushroom(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        
    class_name = classes[pred_idx.item()]
    info_object = mushroom_kb.get(class_name, None)
    
    return class_name, info_object, confidence.item()

# --- RAG Helper Functions ---
def retrieve_from_kb(query, mushroom_kb):
    """Retrieve relevant information from knowledge base"""
    query_lower = query.lower()
    relevant = []
    
    # Check for exact species matches
    for species_name, info in mushroom_kb.items():
        if (species_name.lower() in query_lower or 
            info['common_name'].lower() in query_lower):
            relevant.append((species_name, info))
    
    # If no exact match, check for keyword matches
    if not relevant:
        for species_name, info in mushroom_kb.items():
            # Check if any keywords from description match
            desc_words = set(info['description'].lower().split())
            query_words = set(query_lower.split())
            if desc_words.intersection(query_words):
                relevant.append((species_name, info))
    
    # Return all KB if no matches (for general questions)
    if not relevant:
        relevant = list(mushroom_kb.items())
    
    return relevant[:3]  # Return top 3 most relevant

def format_kb_context(relevant_items):
    """Format KB items as text for LLM - concise version"""
    if not relevant_items:
        return "No specific mushroom information available."
    
    context = ""
    for species, info in relevant_items:
        # More concise format - only essential info
        context += f"{species} ({info['common_name']}): {info['edibility']}. "
        context += f"Key features: {', '.join(info['distinguishing_features'][:2])}. "
        context += f"Safety: {info['safety_warning'][:100]}...\n"
    return context

def get_chat_response_fallback(user_input, mushroom_kb):
    """Fallback rule-based response if LLM fails"""
    user_input_lower = user_input.lower()
    
    # Introduction/help responses
    if any(word in user_input_lower for word in ["hello", "hi", "help", "introduction", "what is this"]):
        return """Hello! I'm a Mushroom Classification Chatbot. I can help you:
        
🍄 **Identify mushrooms** from images - Upload an image using the upload button!
📚 **Learn about mushroom species** - Ask me about any of the 9 genera I know
⚠️ **Understand safety** - I'll warn you about dangerous mushrooms
🔍 **Get identification tips** - Ask about distinguishing features

**Try asking:**
- "Tell me about Agaricus"
- "What is Amanita?"
- "How do I identify mushrooms?"
- "What are the safety warnings?"

Or upload an image to get a classification!"""
    
    if any(word in user_input_lower for word in ["identify", "how to", "tips", "identifying"]):
        return """**Mushroom Identification Tips:**

1. **Spore Print**: Place the cap gill-side down on paper for 1-2 hours to see the spore color
2. **Check the Base**: Look for a volva (cup) at the base of the stem - this is critical for safety
3. **Gill Color**: Note the color of the gills (white, brown, pink, etc.)
4. **Cap Features**: Observe cap shape, color, texture
5. **Stem Features**: Check for rings, color, texture
6. **Habitat**: Note where it's growing (on wood, ground, type of trees nearby)

**⚠️ CRITICAL SAFETY RULE**: Never eat a mushroom based on visual identification alone. Always consult an expert for edibility!"""
    
    # Check for specific mushroom species
    for species_name, info in mushroom_kb.items():
        if species_name.lower() in user_input_lower or info['common_name'].lower() in user_input_lower:
            response = f"**{species_name}** ({info['common_name']})\n\n"
            response += f"**Edibility:** {info['edibility']}\n\n"
            response += f"**Description:** {info['description']}\n\n"
            response += "**Distinguishing Features:**\n"
            for feature in info['distinguishing_features']:
                response += f"- {feature}\n"
            response += f"\n**Family:** {info['taxonomy']['family']}\n\n"
            response += f"**⚠️ Safety Warning:** {info['safety_warning']}"
            return response
    
    # Safety-related questions
    if any(word in user_input_lower for word in ["safe", "edible", "poisonous", "dangerous", "toxic"]):
        return """**Mushroom Safety Guidelines:**

⚠️ **NEVER eat a mushroom unless you are 100% certain of its identity!**

**Deadly Mushrooms to Avoid:**
- **Amanita species** (Death Cap, Destroying Angel) - Often mistaken for edible mushrooms
- Some **Cortinarius** species - Can cause kidney failure
- **Entoloma** species - Many are toxic

**Key Safety Rules:**
1. When in doubt, throw it out
2. Always get a spore print
3. Check for a volva (cup) at the base
4. Consult multiple field guides
5. Consider consulting a mycologist

**This app is for educational purposes only and should NOT be used as the sole method for identifying edible mushrooms.**"""
    
    # General mushroom questions
    if any(word in user_input_lower for word in ["what is", "what are", "mushroom", "tell me about", "tell me more"]):
        return """**Mushrooms** are the fruiting bodies of fungi. They play crucial roles in ecosystems as decomposers.

**Key Mushroom Parts:**
- **Cap** (pileus): The top part
- **Gills** (lamellae): Under the cap, where spores are produced
- **Stem** (stipe): Supports the cap
- **Volva**: A cup-like structure at the base (present in Amanitas)
- **Ring**: A skirt-like structure on the stem

**Spore Print**: The color of spores released by the mushroom - critical for identification!

I can identify 9 common mushroom genera: Agaricus, Amanita, Boletus, Cortinarius, Entoloma, Hygrocybe, Lactarius, Russula, and Suillus. Ask me about any of them!"""
    
    # Default response
    return """I can help you learn about mushrooms! Try asking:
- About a specific species (e.g., "Tell me about Agaricus")
- Identification tips ("How do I identify mushrooms?")
- Safety information ("What are the safety warnings?")
- General questions ("What is a mushroom?")

Or upload an image using the upload button next to the chat input!"""

# --- LLM-Powered Chat Response Function ---
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with API key from secrets"""
    try:
        # Try to get API key from secrets
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            if api_key and api_key.strip():
                client = OpenAI(api_key=api_key)
                # Test the client with a simple call
                return client
            else:
                st.sidebar.error("⚠️ OpenAI API key is empty in secrets.toml")
                return None
        else:
            st.sidebar.error("⚠️ OPENAI_API_KEY not found in secrets. Please check .streamlit/secrets.toml")
            st.sidebar.info("💡 Make sure the file exists at: .streamlit/secrets.toml")
            return None
    except KeyError:
        st.sidebar.error("⚠️ OPENAI_API_KEY not found in secrets")
        return None
    except Exception as e:
        st.sidebar.error(f"⚠️ Error loading OpenAI API key: {e}")
        return None

def get_chat_response(user_input, mushroom_kb):
    """LLM-powered chatbot using RAG with knowledge base"""
    # Initialize OpenAI client
    client = get_openai_client()
    
    # If API key not available, use fallback
    if client is None:
        return get_chat_response_fallback(user_input, mushroom_kb)
    
    # Retrieve relevant KB information (limit to 2 most relevant for speed)
    relevant_kb = retrieve_from_kb(user_input, mushroom_kb)[:2]  # Only top 2 for faster responses
    kb_context = format_kb_context(relevant_kb)
    
    # System prompt following Sporacle persona
    system_prompt = """You are Sporacle, a friendly, safety-first mushroom expert chatbot. Your tagline: "Ask before you snack."

CORE PRINCIPLES:
- Safety over certainty: If identification is not conclusive, explicitly say "Do not eat this."
- No edibility from photos alone: Photos can narrow candidates but do not finalize edibility.
- Explain, then name: Teach the why behind each trait (cap, gills/pores, stipe, spore print, habitat).
- Zero unqualified jargon: If you use a technical term, define it immediately in parentheses. Example: "volva (a cup-like base at the bottom of the stem)"
- Kind refusals: If risk is high, refuse consumption guidance and provide safe next steps.

CONVERSATION STYLE:
- Tone: warm, encouraging teacher; concise and structured
- Format: short sections and checklists
- Define terms inline: always define technical terms in parentheses immediately
- Progressive disclosure: start simple; offer deeper detail if invited
- On uncertainty: give a confidence band (Low / Medium / High) with 1–2 reasons

GOLDEN SAFETY RULES:
- Never claim edibility from a single image or uncertain ID
- If any doubt: "I am not certain—do not consume."
- Present look-alike risks explicitly, especially deadly taxa
- Encourage spore print and base excavation (to check for volva)

You have access to a knowledge base with information about 9 mushroom genera. Use this information to answer questions accurately. Always define technical terms inline and emphasize safety."""

    # Create user message with KB context
    user_message = f"""Knowledge Base Information:
{kb_context}

User Question: {user_input}

Provide a helpful, warm response using the knowledge base information above. Always define technical terms inline (term in parentheses). Emphasize safety and never claim edibility from uncertain IDs. Use confidence bands (Low/Medium/High) when appropriate."""

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o-mini for best value
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,  # Balanced for warm but focused responses
            max_tokens=300  # Allow for structured responses with definitions
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        # Fallback to rule-based if API call fails
        error_msg = str(e)
        # Only show error in sidebar to avoid cluttering chat
        with st.sidebar:
            st.error(f"⚠️ LLM API error: {error_msg}")
        return get_chat_response_fallback(user_input, mushroom_kb)

# --- Format Prediction Response (Sporacle Style) ---
def format_prediction_response(pred_class, info, confidence_score):
    """Format model prediction in Sporacle style with structured output"""
    # Determine confidence band
    if confidence_score >= 0.8:
        confidence_band = "High"
    elif confidence_score >= 0.5:
        confidence_band = "Medium"
    else:
        confidence_band = "Low"
    
    response = "**What I see**\n\n"
    response += f"Model prediction: **{pred_class}** ({confidence_band} confidence: {confidence_score*100:.1f}%)\n\n"
    
    if info:
        # Most likely candidate
        response += f"**Most likely candidate**\n\n"
        response += f"**{pred_class}** ({info['common_name']}) — {confidence_band}: "
        
        # Add key distinguishing features
        key_features = info['distinguishing_features'][:2]
        response += "; ".join(key_features) + ".\n\n"
        
        # Look-alikes & Risks
        response += "**Look-alikes & Risks**\n\n"
        
        # Special case for Agaricus
        if pred_class == "Agaricus":
            response += "⚠️ **Amanita sp. (deadly)** — This looks similar to edible Agaricus, but Amanita species have a volva (a cup-like base at the bottom of the stem) and white spore prints. Agaricus have dark brown spore prints and no volva.\n\n"
        elif pred_class == "Amanita":
            response += "⚠️ **DEADLY** — This genus contains some of the most toxic mushrooms. Never consume any Amanita species.\n\n"
        elif info['edibility'] in ["Deadly Poisonous", "Varies (Some Deadly)", "Varies (Some Toxic)"]:
            response += f"⚠️ **High risk** — {info['safety_warning']}\n\n"
        else:
            response += f"⚠️ **Caution** — {info['safety_warning']}\n\n"
        
        # Next steps
        response += "**Next steps**\n\n"
        response += "1. Do a spore print (place the cap gill-side down on paper for 6-12 hours; the spore print color helps narrow the genus).\n"
        response += "2. Carefully excavate the stipe (stem) base to check for a volva (cup-like base).\n"
        response += "3. Note any bruising (color change after pressing) or distinctive odor.\n\n"
        if mushroom_diagram_path:
            response += "💡 **Tip**: Check the sidebar for a mushroom parts diagram to help identify these features!\n\n"
        
        # Verdict
        response += "**Verdict**\n\n"
        if confidence_band == "Low" or pred_class == "Agaricus":
            response += "⚠️ **Uncertain ID. Do not consume.** Verify with a local expert or share more details (spore print color, stipe base features).\n\n"
        elif info['edibility'] in ["Deadly Poisonous", "Varies (Some Deadly)", "Varies (Some Toxic)"]:
            response += "⚠️ **Do not consume.** This group contains deadly species. Consult an expert.\n\n"
        else:
            response += "⚠️ **I cannot confirm edibility from images alone.** This group has dangerous look-alikes. Please don't eat it. Verify with a local expert.\n\n"
        
        # Mini-lesson
        response += "**Mini-lesson**\n\n"
        if "spore print" in str(info['distinguishing_features']).lower():
            response += "A spore print is the powder color left by spores—it's a key ID clue that helps separate look-alike genera.\n\n"
        elif "volva" in str(info['distinguishing_features']).lower():
            response += "A volva is a cup-like base at the bottom of the stem. Many deadly Amanita species have one, so always dig up the base to check.\n\n"
        else:
            response += f"Key ID trait: {info['distinguishing_features'][0] if info['distinguishing_features'] else 'Check multiple features for safe identification'}.\n\n"
    
    return response

# --- Main UI ---
st.title("🍄 Sporacle")
st.caption("Ask before you snack. A friendly, science-grounded fungi guide.")

# Sidebar for quick reference only
with st.sidebar:
    st.header("📚 Quick Reference")
    st.write("**I can identify these 9 mushroom genera:**")
    for species in mushroom_kb.keys():
        st.write(f"- {species}")
    
    st.markdown("---")
    
    # Display mushroom diagram in sidebar if available
    if mushroom_diagram_path:
        st.header("🍄 Mushroom Parts Guide")
        st.image(mushroom_diagram_path, caption="Mushroom anatomy: cap, gills, stipe (stem), volva, spore print", use_container_width=True)
        st.markdown("**Key terms:**")
        st.markdown("- **Cap**: The top part of the mushroom")
        st.markdown("- **Gills**: The thin structures under the cap (or pores in boletes)")
        st.markdown("- **Stipe**: The stem")
        st.markdown("- **Volva**: A cup-like base at the bottom of the stem (often buried)")
        st.markdown("- **Spore print**: The color of spores left on paper")
        st.markdown("---")
    
    # Debug: Check API key status
    try:
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            if api_key and api_key.strip():
                st.success("✅ OpenAI API key loaded")
            else:
                st.error("❌ OpenAI API key is empty")
        else:
            st.warning("⚠️ OpenAI API key not found in secrets")
    except Exception as e:
        st.error(f"❌ Error checking API key: {e}")
    
    st.markdown("---")
    st.write("**⚠️ Safety Reminder:**")
    st.write("Never eat a mushroom based on this app alone. Always consult an expert for edibility!")
    
    st.markdown("---")
    st.write("**ℹ️ About the Model:**")
    st.write("The model analyzes overall mushroom appearance from a single image. It doesn't examine specific parts like gills, volva, or stipe base separately. That's why Sporacle asks for additional information—these features are critical for safe identification!")

# Initialize chat history
if "messages" not in st.session_state:
    welcome_message = {
        "role": "assistant",
        "content": "Hey! I'm **Sporacle**—your friendly fungi guide. 🍄\n\nI can help you learn mushrooms, identify them responsibly, and avoid harm. Share your region, habitat (soil, wood, lawn), and clear photos of the cap (top/side), underside, and the base of the stem (dug intact). I'll explain everything in simple terms—no jargon without definitions.\n\n**Until we're certain, please don't eat it.**\n\nYou can upload an image using the upload button next to the chat input, or ask me questions about mushrooms!"
    }
    # Add Sporacle image to welcome message if available
    if sporacle_image:
        welcome_message["sporacle_image"] = sporacle_image
    st.session_state.messages = [welcome_message]

# Track if we're waiting for verification answers
if "waiting_for_verification" not in st.session_state:
    st.session_state.waiting_for_verification = False
if "pending_prediction" not in st.session_state:
    st.session_state.pending_prediction = None

# Display chat messages
for message in st.session_state.messages:
    # Use Sporacle avatar for assistant messages
    avatar = sporacle_avatar_path if message["role"] == "assistant" and sporacle_avatar_path else None
    with st.chat_message(message["role"], avatar=avatar):
        # Display Sporacle character GIF for assistant messages (if present)
        if message["role"] == "assistant" and "sporacle_image" in message and message["sporacle_image"] is not None and sporacle_gif_base64:
            gif_html = f'<img src="data:image/gif;base64,{sporacle_gif_base64}" width="200" />'
            st.markdown(gif_html, unsafe_allow_html=True)
        # Display uploaded mushroom images
        if "image" in message and message["image"] is not None:
            st.image(message["image"], caption="Uploaded Image", width=300)
        st.markdown(message["content"])

# Handle verification questions for Agaricus (display after messages)
if st.session_state.waiting_for_verification and st.session_state.pending_prediction:
    with st.chat_message("assistant", avatar=sporacle_avatar_path if sporacle_avatar_path else None):
        st.markdown("**I need some additional information to confirm the identification:**\n\n")
        st.markdown("⚠️ **CRITICAL SAFETY WARNING**: The model predicts **Agaricus**, but this mushroom is visually identical to the **deadly 'Destroying Angel' (Amanita bisporigera)**.\n\n")
        st.markdown("**Look-alikes & Risks**: Amanita species (deadly) can look identical to edible Agaricus. We need to check key features.\n\n")
        
        # Display mushroom diagram if available
        if mushroom_diagram_path:
            st.markdown("**📖 Visual Guide - Mushroom Parts:**")
            st.image(mushroom_diagram_path, caption="Mushroom anatomy diagram showing cap, gills, stipe (stem), volva, and spore print", width=400)
            st.markdown("")
        
        st.markdown("Please answer these questions:\n\n")
        
        spore_color = st.selectbox(
            "1. What color is the spore print? (Place the cap gill-side down on paper for 6-12 hours)",
            ("I don't know", "White", "Dark Brown / Chocolate", "Pink", "Other"),
            key="spore_color"
        )
        
        volva_present = st.radio(
            "2. Is there a volva (a cup-like base at the bottom of the stem) at the very base, possibly under the soil?",
            ("I don't know", "Yes, I see a volva (cup-like base)", "No, there is no volva"),
            key="volva_present"
        )
        
        if st.button("Submit Verification", key="submit_verification"):
            # Process verification (Sporacle style)
            if spore_color == "White" or volva_present == "Yes, I see a volva (cup-like base)":
                verification_result = "**Verdict**\n\n"
                verification_result += "⚠️ **DO NOT EAT. This strongly matches the deadly Amanita.**\n\n"
                verification_result += "Amanitas have a **white** spore print and a **volva** (cup-like base). Agaricus have dark brown spore prints and no volva.\n\n"
                verification_result += "**I am not certain—do not consume.** Please verify with a local mycology expert."
            elif spore_color == "Dark Brown / Chocolate" and volva_present == "No, there is no volva":
                verification_result = "**Verdict**\n\n"
                verification_result += "✅ **This confirms the Agaricus ID (Medium confidence).**\n\n"
                verification_result += "Agaricus mushrooms are characterized by a **dark brown** spore print and **no volva** (no cup-like base).\n\n"
                verification_result += "⚠️ **However, I cannot confirm edibility from images alone.** Always consult an expert before consuming any wild mushroom."
            else:
                verification_result = "**Verdict**\n\n"
                verification_result += "ℹ️ **Incomplete verification (Low confidence).**\n\n"
                verification_result += "Please complete the checks to get a safer recommendation. If you're unsure, **do not consume this mushroom**.\n\n"
                verification_result += "**When in doubt, throw it out!**"
            
            # Add verification result to chat
            verification_message = {
                "role": "assistant",
                "content": verification_result
            }
            # Add Sporacle image to verification message
            if sporacle_image:
                verification_message["sporacle_image"] = sporacle_image
            st.session_state.messages.append(verification_message)
            
            st.session_state.waiting_for_verification = False
            st.rerun()

# Bottom input area with file upload and chat input side by side
col1, col2 = st.columns([1, 4])

with col1:
    uploaded_file = st.file_uploader(
        "📤 Upload",
        type=["jpg", "jpeg", "png"],
        help="Upload a mushroom image",
        label_visibility="collapsed"
    )

with col2:
    # Chat input
    if prompt := st.chat_input("Ask me about mushrooms or upload an image..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response
        response = get_chat_response(prompt, mushroom_kb)
        
        # Add assistant response with Sporacle image
        assistant_message = {"role": "assistant", "content": response}
        if sporacle_image:
            assistant_message["sporacle_image"] = sporacle_image
        st.session_state.messages.append(assistant_message)
        
        st.rerun()

# Handle image upload (moved here to process after display)
if uploaded_file is not None and mushroom_kb:
    # Check if this is a new upload
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        
        # Process the image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Add user message with image
        st.session_state.messages.append({
            "role": "user",
            "content": f"Please identify this mushroom: {uploaded_file.name}",
            "image": image
        })
        
        # Classify
        with st.spinner("Analyzing image..."):
            pred_class, info, confidence = predict_mushroom(image)
        
        # Format response
        response = format_prediction_response(pred_class, info, confidence)
        
        # Store prediction info for follow-up
        st.session_state.pending_prediction = {
            "class": pred_class,
            "info": info,
            "confidence": confidence,
            "image": image
        }
        
        # If Agaricus, set flag for verification
        if pred_class == "Agaricus":
            st.session_state.waiting_for_verification = True
            st.session_state.verification_answers = {}
        else:
            st.session_state.waiting_for_verification = False
        
        # Add assistant response (always include image for context)
        assistant_message = {
            "role": "assistant",
            "content": response,
            "image": image
        }
        # Add Sporacle image to assistant message
        if sporacle_image:
            assistant_message["sporacle_image"] = sporacle_image
        st.session_state.messages.append(assistant_message)
        
        st.rerun()
