# Keras-Xception-model-fine-tun

# 🧠 Xception_Keras (4).ipynb
### Title: Training and Evaluating Xception Model

### Purpose:
This notebook is focused on training a deep learning model using the Xception architecture for image classification tasks.

### Key Components:

*  Imports: Brings in libraries from TensorFlow, Keras, NumPy, Pandas, Matplotlib, and Seaborn.

*  Model Tools: Uses components like layers, optimizers, losses, and callbacks, which indicates that the notebook is defining and training a deep neural network.

*  Architecture Source: Uses tensorflow.keras.applications, meaning it likely uses a pretrained Xception model for transfer learning.

📊 Predict.ipynb (newest version)
### Purpose:
This notebook is for predicting using a pre-trained Xception model, matching the one from Xception_Keras (4).ipynb.

### Key Components:

*  Model Loading: Loads model architecture and weights using model_from_json.

*  Evaluation Metrics: Imports precision, recall, F1-score, and classification reports from sklearn.

### 🔁 Connection Between Them:
Xception_Keras (4).ipynb: Trains the Xception model.

Predict.ipynb: Uses that trained model to evaluate test images.

# Dataset usde in traning model:
## https://www.kaggle.com/datasets/rehabalsaby/yemeni-sign-language
