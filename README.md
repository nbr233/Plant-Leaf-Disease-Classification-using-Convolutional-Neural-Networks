# 🌿 Plant Leaf Disease Classification using Convolutional Neural Networks


An AI-powered solution for the automated detection and classification of diseases in plant leaves using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Technology Stack](#-technology-stack)
- [Installation and Setup](#-installation-and-setup)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Future Work](#-future-work)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 📝 Project Overview

This project aims to tackle the challenge of identifying plant diseases accurately and efficiently. By leveraging deep learning, specifically a **Convolutional Neural Network**, this model analyzes images of plant leaves to classify them into various disease categories or as healthy. The goal is to provide a tool that can help in early disease detection, which is crucial for preventing widespread crop damage and ensuring food security.

---

## 📊 Dataset

The model is trained on the **"Plant Village"** dataset, a publicly available collection of plant leaf images.

- **Source**: [Mendeley Data - Plant Village Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)
- **Contents**: The dataset contains thousands of images across numerous classes, representing different plant species and their associated diseases.
- **Preprocessing**: The notebook automatically downloads, extracts, and splits the data into training (80%), validation (10%), and testing (10%) sets. Image augmentation techniques (rotation, flipping, zooming) are applied to the training data to enhance the model's robustness.

---

## 🤖 Model Architecture

The core of this project is a **Convolutional Neural Network (CNN)** implemented using the Keras API within TensorFlow. The architecture is designed for image classification and includes:

- **Convolutional Layers (`Conv2D`)**: To extract features like edges, textures, and patterns from the leaf images.
- **Activation Functions (`ReLU`)**: To introduce non-linearity.
- **Pooling Layers (`MaxPooling2D`)**: To reduce the dimensionality of the feature maps and make the model more efficient.
- **Dropout Layers**: To prevent overfitting by randomly setting a fraction of input units to 0 during training.
- **Dense (Fully Connected) Layers**: To perform the final classification based on the high-level features learned by the convolutional layers.

---

## 🛠️ Technology Stack

- **Language**: `Python 3.x`
- **Libraries**:
  - `TensorFlow` / `Keras`: For building and training the neural network.
  - `NumPy`: For numerical operations.
  - `Matplotlib`: For data visualization and plotting results.
  - `split-folders`: For splitting the dataset into train, validation, and test sets.

---

## ⚙️ Installation and Setup

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing the libraries listed in the Technology Stack section.)*

---

## 🚀 How to Run

The entire workflow is contained within the Jupyter Notebook (`new_plant_disease (1).ipynb`).

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open the notebook:** Navigate to and open `new_plant_disease (1).ipynb`.
3.  **Execute the cells:** Run the cells sequentially from top to bottom. The notebook handles:
    - Downloading and preparing the dataset.
    - Building the CNN model.
    - Training and validating the model.
    - Evaluating the model on the test set.
    - Saving the final trained model (`.keras` file).
    - Predicting the class of a single test image.

Alternatively, you can open and run the notebook directly in **Google Colab**, which provides a free GPU environment and has the necessary libraries pre-installed.

---

## 📈 Results

After training, the model's performance is evaluated on the unseen test dataset. Key performance indicators include:

- **Accuracy**: The model achieves an accuracy of **[Your Accuracy Here, e.g., 98%]** on the test set.
- **Loss and Accuracy Plots**:
  *(Insert your training/validation loss and accuracy plots here.)*
  ![Training Plots](path/to/your/plots.png)
- **Confusion Matrix**:
  *(Insert your confusion matrix image here to visualize classification performance across different classes.)*
  ![Confusion Matrix](path/to/your/confusion_matrix.png)

---

## 🔮 Future Work

Potential enhancements for this project include:

- **Deployment**: Package the model into a web application (using Flask/Django) or a mobile app for real-world use.
- **Model Optimization**: Experiment with transfer learning using pre-trained models like ResNet50, VGG16, or MobileNet to potentially improve accuracy and reduce training time.
- **Real-Time Detection**: Develop a system that can use a live camera feed to identify diseases in real-time.
- **Expand Dataset**: Incorporate more plant species and disease types to create a more comprehensive diagnostic tool.

---

## 📜 License

N/A

---

## 🙏 Acknowledgments

- A huge thanks to the creators of the **Plant Village Dataset** for making their data publicly available.
- This project was inspired by the need for accessible and automated tools in modern agriculture.
