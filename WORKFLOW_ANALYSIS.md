# Workflow Analysis: Current Decision-Making Process

## Current System Architecture

The current system does **NOT** use a formal MDP (Markov Decision Process). Instead, it uses a **hybrid rule-based and LLM-driven system** with the following workflow:

## 1. Text Question Workflow

```
User Question
    ↓
get_chat_response()
    ↓
[Check: OpenAI API available?]
    ├─ Yes → RAG (retrieve_from_kb) → LLM (GPT-4o-mini) → Response
    └─ No → Rule-based fallback (get_chat_response_fallback) → Response
```

**Decision Points:**
- API availability check (binary decision)
- KB retrieval (top 2-3 most relevant entries)
- LLM response generation (deterministic given prompt)

**No MDP elements:**
- No state transitions
- No action selection based on rewards
- No probabilistic state transitions

## 2. Image Upload Workflow

```
Image Upload
    ↓
predict_mushroom(image) [Uses mushroom_model.pt - ResNet18]
    ↓
CNN Classification (ResNet18)
    ↓
[Prediction Class + Confidence]
    ↓
format_prediction_response()
    ↓
[Check: Is prediction Agaricus?]
    ├─ Yes → Set waiting_for_verification = True → Show verification questions
    └─ No → Display response directly
```

**Decision Points:**
- CNN prediction (deterministic given model weights from `mushroom_model.pt`)
- Confidence threshold (implicit, not formalized)
- Agaricus check (hard-coded rule)

**Current Implementation:**
- `predict_mushroom()` loads model from `mushroom_model.pt`
- Model: `FineTuneResNet18` with ResNet18 backbone
- Output: `(class_name, info_object, confidence)`
- This CNN output becomes the **initial state** for the MDP (if implemented)

**No MDP elements:**
- No sequential decision-making
- No state-based action selection
- No reward optimization
- **However**: The CNN classification step would be the **preprocessing** that generates the initial MDP state $s_0$

## 3. Verification Workflow (Agaricus Case)

```
Agaricus Prediction
    ↓
Show verification questions (spore color, volva presence)
    ↓
User provides answers
    ↓
Rule-based decision:
    if (spore_color == "White" OR volva_present == "Yes"):
        → DANGER (Amanita)
    elif (spore_color == "Dark Brown" AND volva_present == "No"):
        → SAFE (Agaricus confirmed)
    else:
        → UNCERTAIN
```

**Decision Points:**
- Binary rule-based logic
- No probabilistic reasoning
- No sequential information gathering

**No MDP elements:**
- No state space definition
- No action space (questions are fixed)
- No transition probabilities
- No reward function

## Current Limitations

1. **No Sequential Decision-Making**: The system doesn't decide which questions to ask next based on previous answers
2. **No Uncertainty Quantification**: Confidence is computed but not used for decision-making
3. **No Information Value**: All questions are asked regardless of information gain
4. **No Adaptive Strategy**: The verification flow is fixed, not optimized

## How This Could Be Modeled as an MDP

### Complete MDP Formulation Including CNN Classification

The MDP includes the initial CNN classification step using `mushroom_model.pt` as the **preprocessing step** that generates the initial state.

#### Step 0: CNN Classification (Pre-MDP)

**Input:** Raw image $I \in \mathbb{R}^{224 \times 224 \times 3}$

**CNN Forward Pass:**
1. Preprocess: $I' = \text{Transform}(I)$ (resize, normalize)
2. CNN prediction: $P = f_\theta(I')$ where $f_\theta$ is the ResNet18 model from `mushroom_model.pt`
3. Class prediction: $\hat{y} = \arg\max_j P_j$
4. Confidence: $c = \max_j P_j$
5. Knowledge base lookup: $KB(\hat{y})$

**Output:** Initial MDP state $s_0$

#### MDP Formulation (Post-CNN)

**States (S):**
- **Initial state** (from CNN): `s_0 = (predicted_class, confidence, KB_info, image_features)`
  - `predicted_class`: Output from CNN (`mushroom_model.pt`)
  - `confidence`: CNN confidence score $c \in [0, 1]$
  - `KB_info`: Knowledge base information for predicted class
  - `image_features`: Optional - extracted CNN features (before classifier)
  
- **Intermediate states**: `s_t = (predicted_class, confidence, KB_info, features_observed, answers_to_date, uncertainty)`
  - `features_observed`: Set of morphological features observed so far
  - `answers_to_date`: User responses to questions asked
  - `uncertainty`: Updated uncertainty after gathering information
  
- **Terminal states**: `s_T = (final_classification, safety_status, decision_confidence)`
  - `final_classification`: Final identified class (may differ from initial CNN prediction)
  - `safety_status`: SAFE / DANGER / UNCERTAIN
  - `decision_confidence`: Confidence in final decision

**Actions (A):**
- `a_0`: Ask about spore print
- `a_1`: Ask about volva
- `a_2`: Ask about gill color
- `a_3`: Ask about habitat
- `a_4`: Ask about bruising/odor
- `a_5`: Make final decision (SAFE/DANGER/UNCERTAIN)
- `a_6`: Request additional images (cap, gills, base)

**Transition Probabilities (P):**
- `P(s_{t+1} | s_t, a_t)`: Probability of reaching next state given current state and action
- **Example 1**: `P(s_{t+1} | s_t, ask_volva)` depends on user's answer:
  - If volva = "No": `P(high_confidence_agaricus | s_t, ask_volva) = 0.8`
  - If volva = "Yes": `P(danger_amanita | s_t, ask_volva) = 0.9`
  
- **Example 2**: `P(s_{t+1} | s_t, ask_spore_print)`:
  - If spore = "Dark Brown": `P(confirm_agaricus | s_t, ask_spore) = 0.7`
  - If spore = "White": `P(danger_amanita | s_t, ask_spore) = 0.85`

**Reward Function (R):**
- `R(s, a)`: Reward for taking action `a` in state `s`
- **Positive rewards:**
  - Correct identification: $+100$
  - User safety preserved: $+50$
  - High confidence decision: $+10$
  
- **Negative rewards:**
  - Wrong identification (false negative for deadly): $-1000$ (critical!)
  - Wrong identification (false positive for edible): $-100$
  - User harm: $-10000$ (catastrophic)
  
- **Costs:**
  - Asking a question: $-1$ (time/effort)
  - Unnecessary questions: $-2$ (if information doesn't help)

**Policy (π):**
- `π(a | s)`: Probability of taking action `a` in state `s`
- **Optimal policy**: $\pi^* = \arg\max_\pi \mathbb{E}[\sum_{t=0}^T \gamma^t R(s_t, a_t)]$
- **Discount factor**: $\gamma \in [0, 1]$ (typically 0.9-0.99)

### Complete MDP Workflow (Including CNN)

```
[PRE-MDP: CNN Classification]
Image I → mushroom_model.pt (ResNet18) → 
  P = f_θ(I) → 
  ŷ = argmax(P) = "Agaricus", c = 0.75 →
  KB(Agaricus) lookup →
  
[MDP STARTS: Initial State]
s_0 = (predicted_class="Agaricus", confidence=0.75, KB_info, features={})
    ↓
[Action Selection: π(a | s_0)]
    ↓ [Action: a_0 = Ask about spore print]
    
s_1 = (predicted_class="Agaricus", confidence=0.75, 
       features={spore_print="Dark Brown"}, answers={spore="Dark Brown"})
    ↓
[Action Selection: π(a | s_1)]
    ↓ [Action: a_1 = Ask about volva]
    
s_2 = (predicted_class="Agaricus", confidence=0.85, 
       features={spore_print="Dark Brown", volva="No"}, 
       answers={spore="Dark Brown", volva="No"})
    ↓
[Action Selection: π(a | s_2)]
    ↓ [Action: a_5 = Make final decision]
    
[MDP TERMINATES]
s_T = (final_classification="Agaricus", safety_status="SAFE", 
       decision_confidence=0.85)
```

### State Space Size Analysis

**State space dimensions:**
- `predicted_class`: 9 possible values (9 mushroom genera)
- `confidence`: Continuous $[0, 1]$, discretized to 10 bins → 10 values
- `KB_info`: Categorical (fixed per class) → 9 values
- `features_observed`: $2^5 = 32$ combinations (5 possible features)
- `answers_to_date`: $3^5 = 243$ combinations (5 questions, 3 answers each: Yes/No/Unknown)
- `uncertainty`: Continuous $[0, 1]$, discretized to 10 bins → 10 values

**Total state space size (discretized):**
- $|S| = 9 \times 10 \times 9 \times 32 \times 243 \times 10 = 6,298,800$ states
- **Practical reduction**: Many states are unreachable or equivalent
- **Effective state space**: ~$10^4$ to $10^5$ states (after pruning)

### Integration with CNN Model

The MDP is **tightly integrated** with the CNN classification:

1. **CNN provides initial state**: The ResNet18 model (`mushroom_model.pt`) generates $s_0$
2. **CNN confidence informs MDP**: Low confidence → more questions needed
3. **CNN features can be used**: Extracted features (before classifier) can inform state
4. **MDP refines CNN prediction**: MDP can override CNN if evidence contradicts

**Mathematical formulation:**
$$s_0 = \text{CNN\_State}(I) = (\arg\max_j f_\theta(I)_j, \max_j f_\theta(I)_j, KB(\arg\max_j f_\theta(I)_j))$$

Where $f_\theta$ is the ResNet18 model loaded from `mushroom_model.pt`.

## Benefits of Adding MDP

1. **Adaptive Question Selection**: Ask most informative questions first
2. **Uncertainty-Aware Decisions**: Make decisions based on confidence levels
3. **Optimal Information Gathering**: Minimize questions while maximizing safety
4. **Formal Decision Framework**: Mathematically rigorous approach to safety-critical decisions

## Recommendation

To add technical depth, consider implementing an MDP-based decision system for the verification workflow, especially for the Agaricus case. This would:
- Add significant technical depth to the project
- Demonstrate understanding of decision systems (one of the course themes)
- Provide a formal framework for safety-critical decisions
- Allow for analysis of state space size, policy optimization, etc.

