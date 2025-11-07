# Mathematical Formulation: Mushroom Classification System

## 1. Problem Formulation

### 1.1 Supervised Classification Problem

Given a dataset $\mathcal{D} = \{(I_i, y_i)\}_{i=1}^n$ where:
- $I_i \in \mathbb{R}^{224 \times 224 \times 3}$ is a preprocessed RGB image
- $y_i \in \{1, 2, \ldots, 9\}$ is the class label (9 mushroom genera: Agaricus, Amanita, Boletus, Cortinarius, Entoloma, Hygrocybe, Lactarius, Russula, Suillus)

We aim to learn a mapping:
$$f_\theta: \mathbb{R}^{224 \times 224 \times 3} \rightarrow \Delta^9$$

where $\Delta^9 = \{p \in \mathbb{R}^9 : \sum_{j=1}^9 p_j = 1, p_j \geq 0\}$ is the 9-dimensional probability simplex.

### 1.2 Loss Function

The training objective is to minimize the cross-entropy loss:

$$\mathcal{L}(\theta) = -\frac{1}{n}\sum_{i=1}^n \sum_{j=1}^9 \mathbb{1}[y_i = j] \log P(y_i = j | I_i; \theta)$$

where $P(y = j | I; \theta) = \text{softmax}(f_\theta(I))_j$ and $\mathbb{1}[\cdot]$ is the indicator function.

---

## 2. Transfer Learning Formulation

### 2.1 Pretrained Backbone

The model uses a ResNet18 backbone pretrained on ImageNet:
- **Backbone**: $\phi_{\theta_{\text{ImageNet}}}: \mathbb{R}^{224 \times 224 \times 3} \rightarrow \mathbb{R}^{512}$
- **Initialization**: $\theta_{\text{backbone}}^{(0)} = \theta_{\text{ImageNet}}$ (pretrained weights)
- **Feature extraction**: $z = \phi_{\theta_{\text{backbone}}}(I) \in \mathbb{R}^{512}$

### 2.2 Fine-Tuning Objective

The transfer learning objective is:

$$\theta^* = \arg\min_{\theta} \mathcal{L}_{\text{task}}(\theta; \mathcal{D}_{\text{task}})$$

where:
- $\theta = \{\theta_{\text{backbone}}, \theta_{\text{extra}}, \theta_{\text{classifier}}\}$
- $\theta_{\text{backbone}}^{(0)} = \theta_{\text{ImageNet}}$ (initialized from pretrained)
- $\theta_{\text{extra}}^{(0)}, \theta_{\text{classifier}}^{(0)}$ are randomly initialized

### 2.3 Learning Rate Strategy

Different learning rates for different components:
- **Backbone**: $\eta_{\text{backbone}} = \eta \cdot \alpha$ (smaller, e.g., $\alpha = 0.1$)
- **New layers**: $\eta_{\text{extra}} = \eta_{\text{classifier}} = \eta$ (full learning rate)

This allows the pretrained features to adapt slowly while new layers learn quickly.

---

## 3. Model Architecture

### 3.1 Complete Forward Pass

The model $f_\theta$ is composed of three components:

#### Step 1: Feature Extraction (ResNet18 Backbone)
$$z_1 = \phi_{\theta_{\text{backbone}}}(I) \in \mathbb{R}^{512}$$

The ResNet18 backbone consists of:
- Initial conv layer: $7 \times 7$ conv, stride 2
- Max pooling: $3 \times 3$, stride 2
- 4 residual blocks: $[2, 2, 2, 2]$ layers each
- Global average pooling

#### Step 2: Custom Feature Refinement
$$z_1' = \text{reshape}(z_1) \in \mathbb{R}^{512 \times 1 \times 1}$$

The extra layers perform:
$$z_2 = \text{ReLU}(\text{BN}(\text{Conv}_{3 \times 3}(z_1', 512 \rightarrow 256)))$$
$$z_3 = \text{ReLU}(\text{BN}(\text{Conv}_{3 \times 3}(z_2, 256 \rightarrow 128)))$$
$$z_4 = \text{AdaptiveAvgPool2d}(z_3) \in \mathbb{R}^{128 \times 1 \times 1}$$

where:
- $\text{BN}$: Batch Normalization
- $\text{Conv}_{3 \times 3}$: 3×3 convolution with padding=1
- $\text{ReLU}(x) = \max(0, x)$

#### Step 3: Classification
$$z_5 = \text{flatten}(z_4) \in \mathbb{R}^{128}$$
$$\hat{y} = W \cdot z_5 + b \in \mathbb{R}^9$$
$$P(y | I) = \text{softmax}(\hat{y}) = \frac{\exp(\hat{y}_j)}{\sum_{k=1}^9 \exp(\hat{y}_k)}$$

where $W \in \mathbb{R}^{9 \times 128}$ and $b \in \mathbb{R}^9$ are the classifier parameters.

### 3.2 Complete Formulation

$$f_\theta(I) = \text{softmax}\left(W \cdot \text{flatten}\left(\text{AdaptiveAvgPool2d}\left(\text{ExtraLayers}\left(\text{reshape}\left(\phi_{\theta_{\text{backbone}}}(I)\right)\right)\right)\right) + b\right)$$

---

## 4. Training Algorithm

### 4.1 Optimization

Using Adam optimizer with:
- Learning rate: $\eta = 10^{-4}$
- Batch size: $B = 32$
- Number of epochs: $E = 5$

### 4.2 Batch Processing

For each batch $\mathcal{B} = \{(I_i, y_i)\}_{i=1}^B$:

1. **Forward pass**:
   $$\hat{y}_i = f_\theta(I_i) \quad \forall i \in \mathcal{B}$$

2. **Loss computation**:
   $$\mathcal{L}_{\text{batch}} = -\frac{1}{B}\sum_{i=1}^B \log P(y_i | I_i; \theta)$$

3. **Backward pass**:
   $$\nabla_\theta \mathcal{L}_{\text{batch}} = \frac{\partial \mathcal{L}_{\text{batch}}}{\partial \theta}$$

4. **Parameter update** (Adam):
   $$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot \text{Adam}(\nabla_\theta \mathcal{L}_{\text{batch}}, \theta^{(t)})$$

### 4.3 Training Metrics

- **Training accuracy**: $\text{Acc}_{\text{train}} = \frac{1}{n_{\text{train}}} \sum_{i=1}^{n_{\text{train}}} \mathbb{1}[\arg\max_j P(y=j|I_i) = y_i]$
- **Validation accuracy**: $\text{Acc}_{\text{val}} = \frac{1}{n_{\text{val}}} \sum_{i=1}^{n_{\text{val}}} \mathbb{1}[\arg\max_j P(y=j|I_i) = y_i]$

---

## 5. Hybrid System: CNN + Knowledge Base + Rule-Based Reasoning

### 5.1 Neural Prediction Component

$$P_{\text{CNN}}(y | I) = f_\theta(I) \in \Delta^9$$

### 5.2 Knowledge Base Component

The knowledge base $\mathcal{KB}$ is a structured database:
$$\mathcal{KB} = \{KB(c_j)\}_{j=1}^9$$

where $KB(c_j)$ contains:
- Edibility status: $e(c_j) \in \{\text{Edible}, \text{Poisonous}, \text{Deadly}, \ldots\}$
- Safety warnings: $w(c_j)$
- Distinguishing features: $\mathcal{F}(c_j)$
- Taxonomic information: $\text{Tax}(c_j)$

### 5.3 Rule-Based Reasoning Layer

For the critical Agaricus vs. Amanita case:

**Rule 1**: If $P_{\text{CNN}}(\text{Agaricus} | I) > \tau$ (threshold), then:
- Trigger interactive verification
- Collect user inputs: $u = \{u_{\text{spore}}, u_{\text{volva}}\}$

**Rule 2**: Decision function:
$$D(I, u) = \begin{cases}
\text{SAFE} & \text{if } u_{\text{spore}} = \text{Dark Brown} \land u_{\text{volva}} = \text{No} \\
\text{DANGER} & \text{if } u_{\text{spore}} = \text{White} \lor u_{\text{volva}} = \text{Yes} \\
\text{UNCERTAIN} & \text{otherwise}
\end{cases}$$

### 5.4 Combined Decision System

The final decision combines:
1. **Neural prediction**: $\hat{y}_{\text{CNN}} = \arg\max_j P_{\text{CNN}}(y=j | I)$
2. **Confidence**: $c = \max_j P_{\text{CNN}}(y=j | I)$
3. **Knowledge base lookup**: $KB(\hat{y}_{\text{CNN}})$
4. **Rule-based verification**: $D(I, u)$ (if applicable)

**Final output**:
$$\text{Output}(I) = \begin{cases}
(\hat{y}_{\text{CNN}}, KB(\hat{y}_{\text{CNN}}), c, D(I, u)) & \text{if } \hat{y}_{\text{CNN}} = \text{Agaricus} \\
(\hat{y}_{\text{CNN}}, KB(\hat{y}_{\text{CNN}}), c) & \text{otherwise}
\end{cases}$$

---

## 6. Data Preprocessing

### 6.1 Image Transformation

For each image $I_{\text{raw}} \in \mathbb{R}^{H \times W \times 3}$:

1. **Resize**: $I_1 = \text{Resize}(I_{\text{raw}}, (224, 224))$
2. **Normalize**: $I_2 = \frac{I_1 - \mu}{\sigma}$ where:
   - $\mu = [0.485, 0.456, 0.406]$ (ImageNet mean)
   - $\sigma = [0.229, 0.224, 0.225]$ (ImageNet std)
3. **Tensor conversion**: $I = \text{ToTensor}(I_2) \in [0, 1]^{224 \times 224 \times 3}$

### 6.2 Data Split

$$\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}}$$

where:
- $|\mathcal{D}_{\text{train}}| = 0.8 \times |\mathcal{D}| = 5,371$
- $|\mathcal{D}_{\text{val}}| = 0.2 \times |\mathcal{D}| = 1,343$

---

## 7. Complexity Analysis

### 7.1 Model Size

- **ResNet18 backbone**: ~11.2M parameters
- **Extra layers**: 
  - Conv(512→256): $3 \times 3 \times 512 \times 256 = 1,179,648$ params
  - Conv(256→128): $3 \times 3 \times 256 \times 128 = 294,912$ params
  - BatchNorm: $256 \times 2 + 128 \times 2 = 768$ params
- **Classifier**: $128 \times 9 + 9 = 1,161$ params
- **Total**: ~12.7M parameters

### 7.2 Computational Complexity

**Forward pass**:
- ResNet18: $O(H \times W \times C \times K)$ where $K$ is number of operations
- For $224 \times 224 \times 3$ input: ~$O(10^8)$ operations
- Extra layers: $O(512 \times 256 + 256 \times 128) = O(10^5)$ operations
- **Total**: ~$O(10^8)$ operations per image

**Training complexity**:
- Per epoch: $O(n \times \text{forward} + n \times \text{backward})$
- Backward pass: ~2× forward pass
- Total for 5 epochs: $O(5 \times n \times 3 \times \text{forward}) = O(15 \times n \times 10^8)$
- With $n = 5,371$: ~$O(8 \times 10^{12})$ operations

### 7.3 Space Complexity

- **Input**: $224 \times 224 \times 3 \times 4$ bytes = 602,112 bytes ≈ 0.6 MB
- **Model parameters**: $12.7M \times 4$ bytes ≈ 51 MB
- **Activations** (during forward): ~100 MB
- **Total memory**: ~150-200 MB per batch

---

## 8. Inference Formulation

### 8.1 Prediction Function

For a new image $I_{\text{new}}$:

1. **Preprocess**: $I = \text{Transform}(I_{\text{new}})$
2. **Forward pass**: $P = f_\theta(I)$
3. **Prediction**: $\hat{y} = \arg\max_j P_j$
4. **Confidence**: $c = \max_j P_j$
5. **Knowledge lookup**: $KB(\hat{y})$
6. **Safety check**: If $\hat{y} = \text{Agaricus}$, trigger $D(I, u)$

### 8.2 Confidence Threshold

The system uses confidence $c$ to assess prediction reliability:
- High confidence: $c > 0.8$ → Trust prediction
- Medium confidence: $0.5 < c \leq 0.8$ → Show warning
- Low confidence: $c \leq 0.5$ → Request additional verification

---

## 9. Safety-Critical Considerations

### 9.1 False Negative Rate (Critical Metric)

For deadly mushrooms (e.g., Amanita):
$$\text{FNR} = \frac{\text{False Negatives}}{\text{True Positives} + \text{False Negatives}}$$

**Goal**: Minimize FNR for deadly classes (ideally FNR < 0.01)

### 9.2 Cost Function

Define cost matrix $C_{ij}$ where:
- $C_{ij}$ = cost of predicting class $i$ when true class is $j$
- For safety-critical cases: $C_{\text{Edible, Deadly}} \gg C_{\text{Deadly, Edible}}$

**Weighted loss**:
$$\mathcal{L}_{\text{weighted}}(\theta) = -\frac{1}{n}\sum_{i=1}^n \sum_{j=1}^9 C_{y_i, j} \cdot \mathbb{1}[y_i = j] \log P(y_i = j | I_i; \theta)$$

---

## 10. Summary

The system combines:
1. **Deep learning**: Transfer learning with ResNet18 for visual feature extraction
2. **Knowledge representation**: Structured knowledge base with domain expertise
3. **Rule-based reasoning**: Interactive verification for safety-critical cases
4. **Probabilistic modeling**: Softmax output provides uncertainty quantification

This hybrid approach addresses the limitations of pure deep learning in safety-critical applications by incorporating domain knowledge and human-in-the-loop verification.

