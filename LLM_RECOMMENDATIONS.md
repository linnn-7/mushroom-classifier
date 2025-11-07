# LLM Recommendations for Mushroom Classifier Chatbot

## Top Recommendations

### 1. **OpenAI GPT-3.5-turbo** ⭐⭐⭐⭐⭐ (RECOMMENDED)

**Why:**
- ✅ **Cheapest**: ~$0.0015 per 1K tokens (very affordable for student projects)
- ✅ **Easy integration**: Simple API, well-documented
- ✅ **Good performance**: Handles conversational queries well
- ✅ **Fast**: Low latency
- ✅ **Free tier**: $5 free credits for new users

**Cost Estimate:**
- ~$0.002 per conversation (assuming 500 tokens input + 500 tokens output)
- 1000 conversations = ~$2
- Very affordable for a project

**Integration:**
```python
from openai import OpenAI
client = OpenAI(api_key="your-api-key")
```

**Best for:** Student projects, cost-effective, easy setup

---

### 2. **Anthropic Claude 3 Haiku** ⭐⭐⭐⭐

**Why:**
- ✅ **Better safety**: Designed with safety in mind (important for mushroom identification)
- ✅ **Good reasoning**: Better at following instructions
- ✅ **Affordable**: ~$0.25 per 1M input tokens, $1.25 per 1M output tokens
- ✅ **Long context**: 200K tokens

**Cost Estimate:**
- ~$0.003 per conversation
- Slightly more expensive than GPT-3.5

**Best for:** Safety-critical applications, better instruction following

---

### 3. **OpenAI GPT-4o-mini** ⭐⭐⭐⭐

**Why:**
- ✅ **Better than GPT-3.5**: More capable, better reasoning
- ✅ **Still affordable**: ~$0.15 per 1M input tokens, $0.60 per 1M output tokens
- ✅ **Fast**: Optimized version of GPT-4

**Cost Estimate:**
- ~$0.001 per conversation (cheaper than GPT-3.5!)
- Best value for money

**Best for:** Better performance at similar cost

---

### 4. **Open-Source (Local)** ⭐⭐⭐

**Options:**
- **Llama 3.2 3B** (via Ollama) - Free, runs locally
- **Mistral 7B** - Free, good performance
- **Phi-3** - Free, Microsoft's small model

**Why:**
- ✅ **Free**: No API costs
- ✅ **Privacy**: Data stays local
- ❌ **Setup complexity**: Need to install and run locally
- ❌ **Resource intensive**: Requires GPU or good CPU
- ❌ **Slower**: Local inference is slower

**Best for:** Privacy concerns, no budget, willing to set up locally

---

## Recommendation: **GPT-3.5-turbo or GPT-4o-mini**

For a student project, I recommend:
1. **GPT-4o-mini** (best value) - Better performance, similar cost
2. **GPT-3.5-turbo** (most popular) - Well-tested, reliable

---

## Integration Strategy: RAG (Retrieval Augmented Generation)

Since you have a knowledge base, use **RAG** to:
1. Retrieve relevant info from KB based on user query
2. Pass KB context + user query to LLM
3. LLM generates response using KB information

**Benefits:**
- LLM uses your accurate KB data (not hallucinated info)
- Can answer questions beyond KB (general mushroom knowledge)
- More natural conversations
- Safety warnings from KB are preserved

---

## Implementation Example

Here's how to integrate GPT-3.5-turbo with your KB:

```python
from openai import OpenAI
import json

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_chat_response_with_llm(user_input, mushroom_kb):
    # 1. Retrieve relevant KB info
    kb_context = retrieve_from_kb(user_input, mushroom_kb)
    
    # 2. Create system prompt with safety instructions
    system_prompt = """You are a mushroom identification assistant. You help users identify mushrooms and provide safety information.

CRITICAL SAFETY RULES:
- NEVER encourage eating wild mushrooms
- Always emphasize consulting experts
- Highlight deadly mushrooms (Amanita, etc.)
- When uncertain, recommend not consuming

Use the provided knowledge base information to answer questions accurately."""

    # 3. Format KB context
    kb_text = format_kb_context(kb_context)
    
    # 4. Call LLM
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4o-mini"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Knowledge Base:\n{kb_text}\n\nUser Question: {user_input}"}
        ],
        temperature=0.3,  # Lower = more focused, less creative
        max_tokens=500
    )
    
    return response.choices[0].message.content

def retrieve_from_kb(query, mushroom_kb):
    """Simple retrieval: find relevant species"""
    query_lower = query.lower()
    relevant = []
    
    for species, info in mushroom_kb.items():
        if (species.lower() in query_lower or 
            info['common_name'].lower() in query_lower or
            any(word in query_lower for word in info['description'].lower().split())):
            relevant.append((species, info))
    
    return relevant

def format_kb_context(relevant_items):
    """Format KB items as text for LLM"""
    context = ""
    for species, info in relevant_items:
        context += f"\n{species} ({info['common_name']}):\n"
        context += f"Edibility: {info['edibility']}\n"
        context += f"Description: {info['description']}\n"
        context += f"Features: {', '.join(info['distinguishing_features'])}\n"
        context += f"Safety: {info['safety_warning']}\n"
    return context
```

---

## Cost Comparison

| Model | Cost per 1K tokens | Cost per conversation | Free Tier |
|-------|-------------------|---------------------|-----------|
| GPT-3.5-turbo | $0.0015 | ~$0.002 | $5 credits |
| GPT-4o-mini | $0.15/$0.60 | ~$0.001 | $5 credits |
| Claude 3 Haiku | $0.25/$1.25 | ~$0.003 | Limited |
| Local (Llama) | $0 | $0 | Unlimited |

---

## My Recommendation

**Use GPT-4o-mini** because:
1. ✅ Best value (cheaper than GPT-3.5!)
2. ✅ Better performance
3. ✅ Easy integration
4. ✅ Free credits available
5. ✅ Good for safety-critical applications

**Setup Steps:**
1. Get OpenAI API key (free $5 credits)
2. Add `openai` to requirements.txt
3. Implement RAG with your KB
4. Add safety prompts to system message
5. Test with various queries

---

## Safety Considerations

When using LLM for safety-critical applications:

1. **Always include KB data** - Don't rely on LLM knowledge alone
2. **Strong safety prompts** - Emphasize "never eat without expert"
3. **Validate responses** - Check for safety warnings
4. **Fallback to KB** - If LLM fails, use KB directly
5. **Disclaimers** - Always show "educational purposes only"

---

## Next Steps

1. Choose model (I recommend GPT-4o-mini)
2. Get API key
3. Implement RAG integration
4. Test with your KB
5. Add safety prompts
6. Deploy!

Would you like me to help implement the LLM integration?

