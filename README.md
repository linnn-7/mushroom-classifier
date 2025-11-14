# Mushroom Classifier Chatbot

A **Streamlit app** that classifies mushroom species from images using a fine-tuned ResNet18 model with an MDP-based verification system.  
Upload an image and get the predicted species along with detailed mushroom information and safety assessments.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/linnn-7/mushroom-classifier.git
cd mushroom-classifier
```

### 2. Create Virtual Environment

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up OpenAI API Key (Required for Chatbot)

The chatbot requires an OpenAI API key for LLM functionality.

1. **Create the secrets directory:**
   ```bash
   mkdir -p src/.streamlit
   ```

2. **Create the secrets file:**
   
   Create a file at `src/.streamlit/secrets.toml` with the following content:
   
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```
   
   Replace `"your-api-key-here"` with your actual OpenAI API key.
   
   **To get an API key:**
   - Go to https://platform.openai.com/api-keys
   - Sign up or log in
   - Create a new API key
   - Copy the key and paste it in the secrets.toml file

3. **Verify the file location:**
   The file must be at: `src/.streamlit/secrets.toml` (not `.streamlit/secrets.toml` in the root)

   **Note:** The `secrets.toml` file is in `.gitignore` and will not be pushed to GitHub for security.

### 5. Get the Trained Model

You need the trained model file `mushroom_model.pth` in the `src/` directory.

**Option A: Use Pre-trained Model (if available)**
- Download `mushroom_model.pth` and place it in the `src/` directory
- The file should be at: `src/mushroom_model.pth`

**Option B: Train Your Own Model**
1. **Open the Colab notebook:**
   `src/colab_run.ipynb`

2. **The notebook includes steps to:**
   - Fetch the mushroom dataset from Kaggle  
    - Preprocess the data  
    - Train the model using the training script  
    - Save the trained model as `mushroom_model.pth`

3. **After running, download `mushroom_model.pth` and place it in `src/`:**
   `src/mushroom_model.pth`

#### Training Results

- **Total valid images:** 6,714  
- **Training images:** 5,371  
- **Validation images:** 1,343  
- **Classes:** `['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']`  
- **Device used:** `cuda` (GPU)  

**Final Validation Accuracy:** **86.45%**  
**Model Saved As:** `mushroom_model.pth`

#### CNN Ablation Study Results (`nn_ablation_study.py`)

| Experiment | Description | Validation Accuracy |
|:-----------|:-------------|:-------------------:|
| **Baseline** | Augmentation + Dropout + AdamW | 0.8042 |
| **No Augmentation** | Removed image augmentation | 0.8153 |
| **No Dropout** | Removed dropout layers | 0.8176 |
| **SGD Optimizer** | Replaced AdamW with SGD | 0.4296 |


> **Summary:**  
> - Removing **Dropout** or **Augmentation** slightly improved accuracy (~1%)  
> - Replacing **AdamW** with **SGD** caused a major drop in accuracy  

**📄 Ablation summary file:** `ablation_summary.txt`

## Run the Streamlit App

```bash
cd src
streamlit run app.py
```

- Open the URL displayed in the console (e.g., http://localhost:8501) in your browser.
- The app interface allows you to:
  - Upload mushroom images for classification
  - Chat with the Sporacle chatbot about mushrooms
  - Get detailed safety assessments with MDP-based verification

## Troubleshooting

### Issue: "OPENAI_API_KEY not found"
- **Solution:** Make sure `src/.streamlit/secrets.toml` exists and contains your API key
- Check that the file path is exactly: `src/.streamlit/secrets.toml` (not `.streamlit/secrets.toml` in the root)

### Issue: "mushroom_model.pt not found"
- **Solution:** Download or train the model and place it at `src/mushroom_model.pth`

### Issue: Import errors
- **Solution:** Make sure you've activated the virtual environment and installed all dependencies:
  ```bash
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```

### Issue: Port already in use
- **Solution:** Streamlit will automatically try the next available port, or you can specify one:
  ```bash
  streamlit run app.py --server.port 8502
  ```

## Project Structure

```
mushroom-classifier/
├── src/
│   ├── .streamlit/
│   │   └── secrets.toml          ← Your API key goes here
│   ├── app.py                    ← Main Streamlit application
│   ├── mushroom_model.pth         ← Trained model (you need to get this)
│   ├── mdp_system.py             ← MDP implementation
│   ├── knowledge_base.json       ← Mushroom knowledge base
│   ├── colab_run.ipynb           ← Training notebook
│   └── resnet_trainer.py         ← Train CNN model
|   └── ablation_studies.py       ← System level ablation study
|   └── nn_ablation_study.py      ← CNN ablation study
|   ...
├── requirements.txt
└── README.md
```

## Notes

1. **GPU is recommended** for faster training in Colab
2. **Python 3.10+** is required
3. The `secrets.toml` file is in `.gitignore` and will NOT be pushed to GitHub (for security)
4. The app works without the API key, but chatbot features will be limited

