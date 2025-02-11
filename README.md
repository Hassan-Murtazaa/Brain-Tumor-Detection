# Brain Tumor Detection using Deep Learning

## Overview
This project focuses on the **detection of brain tumors** from MRI images using **deep learning** techniques. It utilizes a **Convolutional Neural Network (CNN)** model trained on a dataset of MRI scans labeled as either tumor or non-tumor. The objective is to develop a model that can assist in automated brain tumor detection.

## Dataset
- The dataset consists of MRI images categorized into two classes:
  - **Yes (Tumor present)**
  - **No (No tumor present)**
- Images are preprocessed and resized to **224x224** pixels before training.
- The dataset is loaded from **Google Drive** using Google Colab.

## Technologies Used
- **Python** (NumPy, Pandas, Matplotlib, Seaborn, OpenCV)
- **TensorFlow & Keras** for Deep Learning
- **Scikit-learn** for model evaluation
- **Google Colab** for cloud-based execution

## Data Preprocessing
- **Image Preprocessing:**
  - Resized images to **224x224** pixels
  - Converted BGR images to RGB format
  - Applied grayscale conversion and Gaussian blur
  - Performed noise removal using thresholding and morphological operations
- **Data Augmentation** to improve generalization
- **Splitting Data:**
  - **80%** for training
  - **10%** for validation
  - **10%** for testing
- **Normalization:** Pixel values scaled to range **[0,1]**

## Model Architecture
The CNN model consists of:
1. **Convolutional Layers:** Extract features from images
2. **MaxPooling Layers:** Reduce dimensionality and computation
3. **Flatten Layer:** Convert feature maps into a single vector
4. **Fully Connected Layers:** Classify images into tumor/non-tumor
5. **Dropout Layer:** Prevent overfitting

```python
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

## Training the Model
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Epochs:** 50
- **Callbacks:**
  - **ModelCheckpoint:** Save the best model based on validation loss
  - **EarlyStopping:** Stop training if validation loss stops improving

## Model Evaluation
- **Accuracy:** Evaluated using test dataset
- **Confusion Matrix:** Displayed using Seaborn
- **ROC Curve & AUC Score:** Used to analyze model performance

## Results
- **Confusion Matrix** for performance evaluation
- **ROC Curve** to analyze True Positive Rate vs False Positive Rate
- **Achieved competitive accuracy in brain tumor detection**

## How to Run the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/Hassan-Murtazaa/brain-tumor-detection.git
   cd brain-tumor-detection
   ```
2. Install dependencies:
   ```sh
   pip install tensorflow keras numpy pandas opencv-python matplotlib seaborn scikit-learn
   ```
3. Run the Jupyter Notebook in Google Colab or locally:
   ```sh
   jupyter notebook Brain Tumor Detection.ipynb
   ```

