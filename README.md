# Mushroom Classifier Chatbot

A **Streamlit app** that classifies mushroom species from images using a fine-tuned ResNet18 model.  
Upload an image and get the predicted species along with detailed mushroom information.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/linnn-7/mushroom-classifier.git
cd project
```

2. **Create a virtual environment**
```bash
python -m venv venv
# Activate the environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Colab: Fetch dataset and generate mushroom_model.pt

1. **Open the Colab notebook:**
`src/colab_run.ipynb`

2. **The notebook includes steps to:**

    - Fetch the mushroom dataset from Kaggle.

    - Preprocess the data.

    - Train the model using train_model.py.

    - Save the trained model as mushroom_model.pt.

3. **After running, download mushroom_model.pt and place it in src/:**
`src/mushroom_model.pt`

### Training Results

- **Total valid images:** 6,714  
- **Training images:** 5,371  
- **Validation images:** 1,343  
- **Classes:** `['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']`  
- **Device used:** `cuda` (GPU)  

**Training Performance (5 epochs):**

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|-------|------------|----------------|----------|--------------|
| 1     | 1.2262     | 0.6384         | 0.7577   | 0.8034       |
| 2     | 0.4898     | 0.9013         | 0.6474   | 0.8086       |
| 3     | 0.2213     | 0.9745         | 0.5171   | 0.8474       |
| 4     | 0.1073     | 0.9944         | 0.4747   | 0.8622       |
| 5     | 0.0664     | 0.9972         | 0.5040   | 0.8392       |



## Run the Streamlit App

`streamlit run app.py`

- Open the URL displayed in the console (e.g., http://localhost:8501) in your browser.

- The app interface allows you to upload mushroom images and see the predicted class with information.

## Notes

1. GPU is recommended for faster training in Colab.

2. data/Mushroom is ignored in GitHub to keep the repo small.

3. Works with Python 3.10+.

## References

- Kaggle mushroom dataset: https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images/data

- resnet18: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html