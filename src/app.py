import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import FineTuneResNet18 
import json 
import os 
import base64
from openai import OpenAI
from mdp_system import (
    MDPState, Action, SafetyStatus, TransitionModel, RewardModel, MDPPolicy,
    create_initial_state, make_final_decision
)

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

# Initialize MDP components
@st.cache_resource
def initialize_mdp_system():
    """Initialize MDP system components"""
    transition_model = TransitionModel()
    reward_model = RewardModel()
    policy = MDPPolicy(transition_model, reward_model)
    return transition_model, reward_model, policy

transition_model, reward_model, mdp_policy = initialize_mdp_system()

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
            max_tokens=600  # Allow for complete structured responses with definitions
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
    """Format model prediction in Sporacle style with structured, colorful output"""
    # Determine confidence band
    if confidence_score >= 0.8:
        confidence_band = "High"
        conf_color = "#22c55e"  # Green
    elif confidence_score >= 0.5:
        confidence_band = "Medium"
        conf_color = "#eab308"  # Yellow
    else:
        confidence_band = "Low"
        conf_color = "#f97316"  # Orange
    
    # Start with HTML wrapper for styling
    response = '<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif;">'
    
    # Header: What I see - soft blue
    response += '<h2 style="color: #7c8ba1; font-size: 1.5em; margin-top: 0; margin-bottom: 0.5em; border-bottom: 2px solid #9ca3af; padding-bottom: 0.3em;">🔍 What I See</h2>'
    response += f'<p style="font-size: 1.1em; color: #ffffff; margin: 0.5em 0;">Model prediction: <strong style="color: #ffffff;">{pred_class}</strong> (<span style="color: {conf_color}; font-weight: bold;">{confidence_band}</span> confidence: {confidence_score*100:.1f}%)</p>'
    response += '<br>'

    if info:
        # Most likely candidate - soft purple
        response += '<h2 style="color: #a78bfa; font-size: 1.4em; margin-top: 1em; margin-bottom: 0.5em; border-bottom: 2px solid #c4b5fd; padding-bottom: 0.3em;">🎯 Most Likely Candidate</h2>'
        response += f'<p style="font-size: 1.05em; color: #ffffff; margin: 0.5em 0; line-height: 1.6;"><strong style="color: #ffffff; font-size: 1.1em;">{pred_class}</strong> ({info["common_name"]}) — <span style="color: {conf_color}; font-weight: bold;">{confidence_band}</span>: '
        
        # Add key distinguishing features
        key_features = info['distinguishing_features'][:2]
        response += "; ".join(key_features) + ".</p>"
        response += '<br>'
        
        # Look-alikes & Risks - soft red
        response += '<h2 style="color: #f87171; font-size: 1.4em; margin-top: 1em; margin-bottom: 0.5em; border-bottom: 2px solid #fca5a5; padding-bottom: 0.3em;">⚠️ Look-alikes & Risks</h2>'
        
        # Special case for Agaricus
        if pred_class == "Agaricus":
            response += '<p style="font-size: 1.05em; color: #ffffff; background-color: rgba(254, 242, 242, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #f87171; margin: 0.5em 0; line-height: 1.6;"><strong>⚠️ Amanita sp. (deadly)</strong> — This looks similar to edible Agaricus, but Amanita species have a <strong>volva</strong> (a cup-like base at the bottom of the stem) and <strong>white spore prints</strong>. Agaricus have <strong>dark brown spore prints</strong> and <strong>no volva</strong>.</p>'
        elif pred_class == "Amanita":
            response += '<p style="font-size: 1.1em; color: #ffffff; background-color: rgba(254, 242, 242, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #f87171; margin: 0.5em 0; line-height: 1.6;"><strong>⚠️ DEADLY</strong> — This genus contains some of the most toxic mushrooms. <strong>Never consume any Amanita species.</strong></p>'
        elif info['edibility'] in ["Deadly Poisonous", "Varies (Some Deadly)", "Varies (Some Toxic)"]:
            response += f'<p style="font-size: 1.05em; color: #ffffff; background-color: rgba(254, 242, 242, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #f87171; margin: 0.5em 0; line-height: 1.6;"><strong>⚠️ High risk</strong> — {info["safety_warning"]}</p>'
        else:
            response += f'<p style="font-size: 1.05em; color: #ffffff; background-color: rgba(255, 251, 235, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #fbbf24; margin: 0.5em 0; line-height: 1.6;"><strong>⚠️ Caution</strong> — {info["safety_warning"]}</p>'
        response += '<br>'
        
        # Next steps - soft green
        response += '<h2 style="color: #6ee7b7; font-size: 1.4em; margin-top: 1em; margin-bottom: 0.5em; border-bottom: 2px solid #86efac; padding-bottom: 0.3em;">📋 Next Steps</h2>'
        response += '<ol style="font-size: 1.05em; color: #ffffff; margin: 0.5em 0; padding-left: 1.5em; line-height: 1.8;">'
        response += '<li style="margin-bottom: 0.5em;">Do a <strong>spore print</strong> (place the cap gill-side down on paper for 6-12 hours; the spore print color helps narrow the genus).</li>'
        response += '<li style="margin-bottom: 0.5em;">Carefully excavate the <strong>stipe (stem) base</strong> to check for a <strong>volva</strong> (cup-like base).</li>'
        response += '<li style="margin-bottom: 0.5em;">Note any <strong>bruising</strong> (color change after pressing) or distinctive <strong>odor</strong>.</li>'
        response += '</ol>'
        if mushroom_diagram_path:
            response += '<p style="font-size: 1em; color: #ffffff; background-color: rgba(238, 242, 255, 0.2); padding: 0.6em; border-radius: 6px; margin: 0.8em 0;">💡 <strong>Tip:</strong> Check the sidebar for a mushroom parts diagram to help identify these features!</p>'
        response += '<br>'
        
        # Verdict - soft brown
        response += '<h2 style="color: #d4a574; font-size: 1.4em; margin-top: 1em; margin-bottom: 0.5em; border-bottom: 2px solid #e5b887; padding-bottom: 0.3em;">⚖️ Verdict</h2>'
        if confidence_band == "Low" or pred_class == "Agaricus":
            response += '<p style="font-size: 1.1em; color: #ffffff; background-color: rgba(254, 242, 242, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #f87171; margin: 0.5em 0; line-height: 1.6;"><strong>⚠️ Uncertain ID. Do not consume.</strong> Verify with a local expert or share more details (spore print color, stipe base features).</p>'
        elif info['edibility'] in ["Deadly Poisonous", "Varies (Some Deadly)", "Varies (Some Toxic)"]:
            response += '<p style="font-size: 1.1em; color: #ffffff; background-color: rgba(254, 242, 242, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #f87171; margin: 0.5em 0; line-height: 1.6;"><strong>⚠️ Do not consume.</strong> This group contains deadly species. Consult an expert.</p>'
        else:
            response += '<p style="font-size: 1.1em; color: #ffffff; background-color: rgba(255, 251, 235, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #fbbf24; margin: 0.5em 0; line-height: 1.6;"><strong>⚠️ I cannot confirm edibility from images alone.</strong> This group has dangerous look-alikes. Please don\'t eat it. Verify with a local expert.</p>'
        response += '<br>'
        
        # Mini-lesson - soft blue
        response += '<h2 style="color: #7dd3fc; font-size: 1.4em; margin-top: 1em; margin-bottom: 0.5em; border-bottom: 2px solid #93c5fd; padding-bottom: 0.3em;">📚 Mini-Lesson</h2>'
        if "spore print" in str(info['distinguishing_features']).lower():
            response += '<p style="font-size: 1.05em; color: #ffffff; background-color: rgba(239, 246, 255, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #7dd3fc; margin: 0.5em 0; line-height: 1.6;">A <strong>spore print</strong> is the powder color left by spores—it\'s a key ID clue that helps separate look-alike genera.</p>'
        elif "volva" in str(info['distinguishing_features']).lower():
            response += '<p style="font-size: 1.05em; color: #ffffff; background-color: rgba(239, 246, 255, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #7dd3fc; margin: 0.5em 0; line-height: 1.6;">A <strong>volva</strong> is a cup-like base at the bottom of the stem. Many deadly Amanita species have one, so always dig up the base to check.</p>'
        else:
            lesson_text = info['distinguishing_features'][0] if info['distinguishing_features'] else 'Check multiple features for safe identification'
            response += f'<p style="font-size: 1.05em; color: #ffffff; background-color: rgba(239, 246, 255, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #7dd3fc; margin: 0.5em 0; line-height: 1.6;">Key ID trait: <strong>{lesson_text}</strong>.</p>'
    
    response += '</div>'
    return response

# Helper function to get answer options for each action
def get_action_options(action: Action) -> list:
    """Get dropdown options for each action"""
    options = {
        Action.ASK_VOLVA: ["I don't know", "Yes", "No"],
        Action.ASK_SPORE_PRINT: ["I don't know", "White", "Dark Brown / Chocolate", "Pink", "Black"],
        Action.ASK_GILL_COLOR: ["I don't know", "White", "Pink", "Brown", "Black", "Yellow"],
        Action.ASK_HABITAT: ["I don't know", "On wood", "On ground/soil"],
        Action.ASK_BRUISING: ["I don't know", "Yes, it changes color", "No, no color change"],
        Action.ASK_ODOR: ["I don't know", "Anise/licorice", "Almond", "Foul/rotten", "No distinct odor"],
    }
    return options.get(action, ["I don't know"])

# Helper function to format checklist
def format_checklist(state: MDPState) -> str:
    """Format checklist with colorful, educational styling"""
    all_features = {
        "ask_volva": "Volva (cup-like base)",
        "ask_spore_print": "Spore print color",
        "ask_habitat": "Habitat",
        "ask_gill_color": "Gill color",
        "ask_bruising": "Bruising",
        "ask_odor": "Odor"
    }
    
    checklist = '<div style="background-color: rgba(240, 249, 255, 0.2); padding: 1em; border-radius: 10px; border: 2px solid #7c8ba1; margin: 1em 0;">'
    checklist += '<h3 style="color: #7c8ba1; font-size: 1.3em; margin-top: 0; margin-bottom: 0.8em; border-bottom: 2px solid #9ca3af; padding-bottom: 0.3em;">📋 Verification Checklist</h3>'
    checklist += '<ul style="list-style: none; padding-left: 0; margin: 0;">'
    
    for feature_key, feature_name in all_features.items():
        if feature_key in state.features_observed:
            answer = state.answers.get(feature_key, "Unknown")
            checklist += f'<li style="margin: 0.6em 0; padding: 0.5em; background-color: rgba(220, 252, 231, 0.2); border-left: 4px solid #6ee7b7; border-radius: 4px;"><span style="color: #6ee7b7; font-size: 1.2em; margin-right: 0.5em;">✅</span><strong style="color: #ffffff;">{feature_name}</strong>: <span style="color: #ffffff;">{answer}</span></li>'
        else:
            checklist += f'<li style="margin: 0.6em 0; padding: 0.5em; background-color: rgba(254, 243, 199, 0.2); border-left: 4px solid #fbbf24; border-radius: 4px;"><span style="color: #fbbf24; font-size: 1.2em; margin-right: 0.5em;">⏳</span><strong style="color: #ffffff;">{feature_name}</strong>: <span style="color: #d1d5db; font-style: italic;">Not yet checked</span></li>'
    
    checklist += '</ul></div>'
    return checklist

# Helper function to format field evaluation
def format_field_evaluation(state: MDPState, final_state: MDPState) -> str:
    """Format evaluation of how each field supports the final verdict"""
    pred_class = state.predicted_class
    evaluation = '<div style="background-color: rgba(239, 246, 255, 0.2); padding: 1em; border-radius: 10px; border: 2px solid #7dd3fc; margin: 1em 0;">'
    evaluation += '<h3 style="color: #7dd3fc; font-size: 1.3em; margin-top: 0; margin-bottom: 0.8em; border-bottom: 2px solid #93c5fd; padding-bottom: 0.3em;">🔬 Field-by-Field Evaluation</h3>'
    
    # Feature names mapping
    feature_names = {
        "ask_volva": "Volva (cup-like base)",
        "ask_spore_print": "Spore print color",
        "ask_habitat": "Habitat",
        "ask_gill_color": "Gill color",
        "ask_bruising": "Bruising",
        "ask_odor": "Odor"
    }
    
    # Evaluate each answered field
    for feature_key, feature_name in feature_names.items():
        if feature_key in state.features_observed:
            answer = state.answers.get(feature_key, "Unknown")
            
            # Generate evaluation based on feature and answer
            eval_text = ""
            eval_color = "#ffffff"
            
            if feature_key == "ask_volva":
                if pred_class == "Agaricus":
                    if answer == "Yes":
                        eval_text = "⚠️ <strong>Critical danger signal:</strong> A volva indicates this is likely an <strong>Amanita</strong> species, not Agaricus. Amanita species are often deadly."
                        eval_color = "#f87171"
                    elif answer == "No":
                        eval_text = "✅ <strong>Supports Agaricus ID:</strong> No volva is consistent with Agaricus species, which helps distinguish from deadly Amanita."
                        eval_color = "#6ee7b7"
                    else:
                        eval_text = "ℹ️ <strong>Uncertain:</strong> Without checking for a volva, we cannot rule out Amanita species."
                        eval_color = "#fbbf24"
                else:
                    if answer == "Yes":
                        eval_text = "ℹ️ <strong>Important feature:</strong> Presence of a volva is a key identifying characteristic for many mushroom genera."
                    elif answer == "No":
                        eval_text = "ℹ️ <strong>Important feature:</strong> Absence of a volva helps narrow down the identification."
                    else:
                        eval_text = "ℹ️ <strong>Uncertain:</strong> Volva presence/absence would help with identification."
            
            elif feature_key == "ask_spore_print":
                if pred_class == "Agaricus":
                    if answer == "White":
                        eval_text = "⚠️ <strong>Critical danger signal:</strong> White spore print indicates this is likely an <strong>Amanita</strong> species, not Agaricus. Agaricus have dark brown spore prints."
                        eval_color = "#f87171"
                    elif answer == "Dark Brown / Chocolate":
                        eval_text = "✅ <strong>Strongly supports Agaricus ID:</strong> Dark brown/chocolate spore print is characteristic of Agaricus species and helps distinguish from deadly Amanita."
                        eval_color = "#6ee7b7"
                    else:
                        eval_text = "ℹ️ <strong>Uncertain:</strong> Spore print color is critical for distinguishing Agaricus from Amanita. Without it, identification is less certain."
                        eval_color = "#fbbf24"
                else:
                    eval_text = f"ℹ️ <strong>Important ID feature:</strong> Spore print color ({answer}) helps narrow down the genus identification."
            
            elif feature_key == "ask_habitat":
                if answer != "I don't know":
                    if pred_class == "Agaricus":
                        if answer == "On ground/soil":
                            eval_text = "✅ <strong>Strongly supports Agaricus ID:</strong> Agaricus species typically grow on soil, lawns, and meadows. This habitat is characteristic and helps distinguish from wood-growing species."
                            eval_color = "#6ee7b7"
                        else:
                            eval_text = "⚠️ <strong>Unusual for Agaricus:</strong> Agaricus species typically grow on soil/ground, not wood. This may indicate a different genus or requires further verification."
                            eval_color = "#fbbf24"
                    elif pred_class in ["Boletus", "Suillus"]:
                        if answer == "On ground/soil":
                            eval_text = f"✅ <strong>Consistent with {pred_class}:</strong> {pred_class} species grow on the ground, often near trees. This habitat supports the identification."
                            eval_color = "#6ee7b7"
                        else:
                            eval_text = f"⚠️ <strong>Unusual for {pred_class}:</strong> {pred_class} species typically grow on the ground near trees, not directly on wood. This may indicate a different genus."
                            eval_color = "#fbbf24"
                    else:
                        eval_text = f"ℹ️ <strong>Supporting evidence:</strong> Growing {answer.lower()} is consistent with {pred_class} species and helps narrow down the identification. Habitat is a key ecological indicator."
                        eval_color = "#7dd3fc"
                else:
                    eval_text = "ℹ️ <strong>Uncertain:</strong> Habitat information is crucial for identification. Many mushroom genera are strongly associated with specific habitats (soil vs. wood), which helps distinguish similar-looking species."
                    eval_color = "#fbbf24"
            
            elif feature_key == "ask_gill_color":
                if answer != "I don't know":
                    if pred_class == "Agaricus":
                        if answer == "Pink":
                            eval_text = "✅ <strong>Strongly supports Agaricus ID:</strong> Pink gills are characteristic of young Agaricus species. As they mature, the gills turn dark brown, which is a key distinguishing feature from deadly Amanita (which have white gills)."
                            eval_color = "#6ee7b7"
                        elif answer == "Brown" or answer == "Dark Brown / Chocolate":
                            eval_text = "✅ <strong>Strongly supports Agaricus ID:</strong> Dark brown gills are characteristic of mature Agaricus species. This is a critical feature that helps distinguish from deadly Amanita species, which have white gills."
                            eval_color = "#6ee7b7"
                        elif answer == "White":
                            eval_text = "⚠️ <strong>Critical danger signal:</strong> White gills indicate this is likely an <strong>Amanita</strong> species, not Agaricus. Agaricus have pink (young) or dark brown (mature) gills. Amanita species are often deadly."
                            eval_color = "#f87171"
                        else:
                            eval_text = f"ℹ️ <strong>Important feature:</strong> {answer} gills are not typical for Agaricus (which have pink or dark brown gills). This may indicate a different genus and requires careful verification."
                            eval_color = "#fbbf24"
                    else:
                        eval_text = f"ℹ️ <strong>Supporting evidence:</strong> {answer} gills are a key identifying feature for {pred_class} species. Gill color, along with spore print color, helps distinguish between similar genera and is essential for accurate identification."
                        eval_color = "#7dd3fc"
                else:
                    eval_text = "ℹ️ <strong>Uncertain:</strong> Gill color is one of the most important identifying features. It helps distinguish between genera (e.g., white gills in Amanita vs. pink/brown in Agaricus) and is critical for safe identification."
                    eval_color = "#fbbf24"
            
            elif feature_key == "ask_bruising":
                if answer != "I don't know":
                    if answer == "Yes, it changes color":
                        if pred_class == "Boletus":
                            eval_text = "✅ <strong>Strongly supports Boletus ID:</strong> Many Boletus species exhibit color changes (often blue or green) when bruised or cut. This 'bluing' reaction is a key identifying characteristic for boletes and helps distinguish them from other genera."
                            eval_color = "#6ee7b7"
                        else:
                            eval_text = f"ℹ️ <strong>Important identifying feature:</strong> Color change when bruised is a characteristic that helps distinguish {pred_class} from similar species. This reaction (often blue, green, or red) is a key diagnostic feature for many mushroom genera."
                            eval_color = "#7dd3fc"
                    else:
                        if pred_class == "Boletus":
                            eval_text = "ℹ️ <strong>Note:</strong> Many Boletus species do change color when bruised, but not all. The absence of color change doesn't rule out Boletus, but it may indicate a different species or genus."
                            eval_color = "#fbbf24"
                        else:
                            eval_text = f"ℹ️ <strong>Supporting evidence:</strong> No color change when bruised is consistent with some {pred_class} species. Bruising behavior (or lack thereof) is an important diagnostic feature that helps narrow down identification."
                            eval_color = "#7dd3fc"
                else:
                    eval_text = "ℹ️ <strong>Uncertain:</strong> Bruising behavior (whether the mushroom changes color when cut or pressed) is a key diagnostic feature, especially for boletes. This simple test can help distinguish between similar-looking species."
                    eval_color = "#fbbf24"
            
            elif feature_key == "ask_odor":
                if answer != "I don't know":
                    if pred_class == "Agaricus":
                        if answer == "Anise/licorice":
                            eval_text = "✅ <strong>Strongly supports Agaricus ID:</strong> Anise or licorice odor is characteristic of some Agaricus species (like Agaricus arvensis). This distinctive smell is a key identifying feature that helps confirm the genus."
                            eval_color = "#6ee7b7"
                        elif answer == "Almond":
                            eval_text = "✅ <strong>Supports Agaricus ID:</strong> Almond-like odor is found in some Agaricus species. This distinctive smell helps distinguish them from other genera and is a useful identifying characteristic."
                            eval_color = "#6ee7b7"
                        elif answer == "Foul/rotten":
                            eval_text = "⚠️ <strong>Unusual for Agaricus:</strong> Foul or rotten odors are not typical for Agaricus species. This may indicate a different genus (like some Russula or Lactarius species) and requires careful verification."
                            eval_color = "#fbbf24"
                        else:
                            eval_text = f"ℹ️ <strong>Supporting evidence:</strong> {answer} odor is a characteristic that helps distinguish {pred_class} from similar species. Odor is an important diagnostic feature, though it can vary between species within a genus."
                            eval_color = "#7dd3fc"
                    else:
                        eval_text = f"ℹ️ <strong>Supporting evidence:</strong> {answer} odor is a distinctive characteristic that helps identify {pred_class} species. Odor can be a key diagnostic feature, especially when combined with other characteristics like gill color and habitat."
                        eval_color = "#7dd3fc"
                else:
                    eval_text = "ℹ️ <strong>Uncertain:</strong> Odor is an important identifying feature for many mushrooms. Some species have distinctive smells (anise, almond, foul, etc.) that can help distinguish them from similar-looking species. Always smell the mushroom carefully (but don't inhale deeply)."
                    eval_color = "#fbbf24"
            
            evaluation += f'<div style="margin: 0.8em 0; padding: 0.8em; background-color: rgba(255, 255, 255, 0.1); border-left: 4px solid {eval_color}; border-radius: 4px;">'
            evaluation += f'<p style="font-size: 1.05em; color: #ffffff; margin: 0.3em 0; line-height: 1.6;"><strong style="color: {eval_color};">{feature_name}:</strong> {answer}</p>'
            evaluation += f'<p style="font-size: 1em; color: {eval_color}; margin: 0.3em 0; line-height: 1.5;">{eval_text}</p>'
            evaluation += '</div>'
    
    evaluation += '</div>'
    return evaluation

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

# Track MDP state for image classification
if "mdp_state" not in st.session_state:
    st.session_state.mdp_state = None
if "current_action" not in st.session_state:
    st.session_state.current_action = None
if "waiting_for_verification" not in st.session_state:
    st.session_state.waiting_for_verification = False
if "pending_prediction" not in st.session_state:
    st.session_state.pending_prediction = None

# Display chat messages
for i, message in enumerate(st.session_state.messages):
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
        # Use HTML rendering for assistant messages to support rich formatting
        if message["role"] == "assistant" and ("<div" in message["content"] or "<h2" in message["content"] or "<h3" in message["content"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])
        
        # If this is an MDP question message, show form with all questions (only for the last one)
        if (message.get("role") == "assistant" and 
            message.get("mdp_question") and 
            st.session_state.mdp_state and 
            not st.session_state.mdp_state.is_terminal and
            i == len(st.session_state.messages) - 1):  # Only show form for the last message
            
            # Add form with all questions at once (use unique key with message index)
            with st.form(key=f"mdp_form_all_questions_{i}", clear_on_submit=False):
                st.markdown("**Please fill in all the information you can:**")
                
                # All possible questions
                all_actions = [
                    (Action.ASK_VOLVA, "Is there a volva (cup-like base) at the stem base, possibly under the soil?"),
                    (Action.ASK_SPORE_PRINT, "What color is the spore print? (Place the cap gill-side down on paper for 6-12 hours)"),
                    (Action.ASK_HABITAT, "Where is the mushroom growing?"),
                    (Action.ASK_GILL_COLOR, "What color are the gills?"),
                    (Action.ASK_BRUISING, "Does the mushroom change color when bruised?"),
                    (Action.ASK_ODOR, "What does the mushroom smell like?"),
                ]
                
                # Store answers in session state
                answers = {}
                for action, question_text in all_actions:
                    options = get_action_options(action)
                    # Get default value (already answered or "I don't know")
                    default_idx = 0  # "I don't know" is first
                    if action.value in st.session_state.mdp_state.features_observed:
                        current_answer = st.session_state.mdp_state.answers.get(action.value, "I don't know")
                        if current_answer in options:
                            default_idx = options.index(current_answer)
                    
                    selected = st.selectbox(
                        question_text,
                        options,
                        key=f"mdp_select_{action.value}",
                        index=default_idx
                    )
                    answers[action.value] = selected
                
                # Submit button
                submitted = st.form_submit_button("Submit All Answers", type="primary", use_container_width=True)
                
                if submitted:
                    # Update MDP state with all answers
                    current_state = st.session_state.mdp_state
                    
                    # Process each answer
                    for action_value, answer in answers.items():
                        # Find the action enum
                        action_enum = None
                        for action in Action:
                            if action.value == action_value:
                                action_enum = action
                                break
                        
                        if action_enum and action_enum != Action.MAKE_DECISION:
                            # Update state with this answer
                            current_state, prob = transition_model.get_transition(
                                current_state,
                                action_enum,
                                answer
                            )
                    
                    # Update session state
                    st.session_state.mdp_state = current_state
                    st.session_state.current_action = None
                    
                    # Clear question key so form doesn't show again
                    if "mdp_question_shown" in st.session_state:
                        del st.session_state["mdp_question_shown"]
                    
                    # After submitting all answers, always make final decision
                    # Make final decision
                    final_state = make_final_decision(current_state)
                    st.session_state.mdp_state = final_state
                    st.session_state.current_action = None
                    
                    # Format final decision
                    result = '<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif;">'
                    result += '<h2 style="color: #d4a574; font-size: 1.6em; margin-top: 0; margin-bottom: 0.8em; border-bottom: 3px solid #e5b887; padding-bottom: 0.3em;">⚖️ Final Verdict</h2>'
                    result += format_checklist(final_state)
                    
                    # Add field-by-field evaluation
                    result += format_field_evaluation(current_state, final_state)
                    result += '<br>'
                    
                    # Final decision
                    if final_state.safety_status == SafetyStatus.DANGER:
                        result += '<p style="font-size: 1.2em; color: #ffffff; background-color: rgba(254, 242, 242, 0.2); padding: 1em; border-radius: 10px; border: 3px solid #f87171; margin: 1em 0; line-height: 1.6;"><strong>⚠️ DO NOT EAT. This mushroom is dangerous.</strong></p>'
                        if current_state.predicted_class == "Agaricus":
                            result += '<p style="font-size: 1.05em; color: #ffffff; margin: 0.8em 0; line-height: 1.6;">Based on the features checked, this strongly matches the deadly <strong>Amanita</strong> species.</p>'
                        result += '<p style="font-size: 1.05em; color: #ffffff; margin: 0.8em 0; line-height: 1.6;"><strong>I am not certain—do not consume.</strong> Please verify with a local mycology expert.</p>'
                    elif final_state.safety_status == SafetyStatus.SAFE:
                        result += f'<p style="font-size: 1.2em; color: #ffffff; background-color: rgba(220, 252, 231, 0.2); padding: 1em; border-radius: 10px; border: 3px solid #6ee7b7; margin: 1em 0; line-height: 1.6;"><strong>✅ This confirms the {current_state.predicted_class} ID</strong> (Confidence: {final_state.decision_confidence*100:.1f}%).</p>'
                        result += '<p style="font-size: 1.05em; color: #ffffff; background-color: rgba(255, 251, 235, 0.2); padding: 0.8em; border-radius: 8px; border-left: 4px solid #fbbf24; margin: 0.8em 0; line-height: 1.6;"><strong>⚠️ However, I cannot confirm edibility from images alone.</strong> Always consult an expert before consuming any wild mushroom.</p>'
                    else:
                        result += '<p style="font-size: 1.1em; color: #ffffff; background-color: rgba(255, 251, 235, 0.2); padding: 1em; border-radius: 10px; border: 3px solid #fbbf24; margin: 1em 0; line-height: 1.6;"><strong>ℹ️ Uncertain ID (Low confidence).</strong></p>'
                        result += '<p style="font-size: 1.05em; color: #ffffff; margin: 0.8em 0; line-height: 1.6;">Please complete additional checks or consult an expert. If you\'re unsure, <strong>do not consume this mushroom</strong>.</p>'
                        result += '<p style="font-size: 1.1em; color: #ffffff; font-weight: bold; margin: 0.8em 0; line-height: 1.6;">When in doubt, throw it out!</p>'
                    
                    result += '</div>'
                    
                    # Add verdict to chat
                    assistant_message = {"role": "assistant", "content": result}
                    if sporacle_image:
                        assistant_message["sporacle_image"] = sporacle_image
                    st.session_state.messages.append(assistant_message)
                    
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
        
        # Get response (normal chat - MDP questions handled via form inputs)
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
        
        # Classify using CNN
        with st.spinner("Analyzing image..."):
            pred_class, info, confidence = predict_mushroom(image)
        
        # Create initial MDP state from CNN classification
        initial_state = create_initial_state(pred_class, confidence, info or {})
        
        # Format initial response
        response = format_prediction_response(pred_class, info, confidence)
        
        # Store prediction info
        st.session_state.pending_prediction = {
            "class": pred_class,
            "info": info,
            "confidence": confidence,
            "image": image
        }
        
        # Initialize MDP state
        st.session_state.mdp_state = initial_state
        
        # Select first action to ask
        first_action = mdp_policy.select_action(initial_state)
        st.session_state.current_action = first_action
        
        # Determine if we need verification (use MDP for all cases, but prioritize Agaricus)
        if pred_class == "Agaricus" or confidence < 0.8:
            st.session_state.waiting_for_verification = True
        else:
            # For high confidence non-Agaricus, still use MDP but may terminate early
            st.session_state.waiting_for_verification = True
        
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

# Add MDP verification question to chat if needed
if st.session_state.mdp_state and not st.session_state.mdp_state.is_terminal:
    # Check if we've already shown the form
    if "mdp_question_shown" not in st.session_state:
        # Format question message
        question_message = '<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif;">'
        question_message += '<h2 style="color: #a78bfa; font-size: 1.5em; margin-top: 0; margin-bottom: 0.8em; border-bottom: 2px solid #c4b5fd; padding-bottom: 0.3em;">🔍 Additional Information Needed</h2>'
        question_message += '<p style="font-size: 1.05em; color: #ffffff; margin: 0.5em 0; line-height: 1.6;">I need some additional information to confirm the identification. Please fill in all the information you can:</p>'
        question_message += '</div>'
        
        # Add question to chat
        assistant_message = {
            "role": "assistant",
            "content": question_message,
            "mdp_question": True
        }
        if sporacle_image:
            assistant_message["sporacle_image"] = sporacle_image
        st.session_state.messages.append(assistant_message)
        st.session_state["mdp_question_shown"] = True
        st.rerun()
