# **Pneumonia Detection Using InceptionV3**

This repository contains the code and resources for detecting pneumonia from chest X-ray images using the InceptionV3 deep learning model. The project leverages PyTorch for model development and training.

![image](https://github.com/Abhaykumar04/Pneumonia-Detection-Using-InceptionV3/assets/112232080/8680048f-a654-4645-806c-754e9acaa26b)

---

**Table of Contents**

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

### **Introduction**

Pneumonia is a serious lung infection that can be life-threatening. Early detection is crucial for effective treatment. This project aims to automate the detection of pneumonia from chest X-ray images using a pre-trained deep learning model (InceptionV3), which has been fine-tuned for this specific task.

![image](https://github.com/Abhaykumar04/Pneumonia-Detection-Using-InceptionV3/assets/112232080/e5c97d15-4141-4372-a356-28841cb38948)


---

### **Dataset**

- **Source:** The dataset used in this project is the Chest X-Ray Images (Pneumonia) dataset, available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
- **Structure:** The dataset is organized into three directories: `train`, `val`, and `test`, each containing subdirectories for `NORMAL` and `PNEUMONIA` images.
- **Size:** The dataset comprises thousands of X-ray images labeled as either `NORMAL` or `PNEUMONIA`.

---

### **Model**

- **Architecture:** InceptionV3, a convolutional neural network known for its deep architecture and efficient computation.
- **Modifications:** The final fully connected layer is modified to output two classes (normal and pneumonia).
- **Training:** The model is trained using the Adam optimizer and Cross-Entropy Loss, with data augmentation applied to improve generalization.

---

### **Prerequisites**

- Python 3.7 or higher
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-image
- scikit-learn

---

### **Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/pneumonia-detection-using-inceptionv3.git
   cd pneumonia-detection-using-inceptionv3
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   - Download the dataset from Kaggle and place it in the appropriate directory as specified in the notebook.

---

### **Usage**

1. **Train the model:**
   - Open the Jupyter notebook `pneumonia-detection-using-inceptionv3.ipynb`.
   - Run all cells to train the model on the provided dataset.

2. **Evaluate the model:**
   - The notebook includes code to evaluate the trained model on the test dataset and visualize the results.

3. **Inference:**
   - Use the trained model to predict pneumonia in new chest X-ray images by modifying the inference section in the notebook.

---

### **Results**

- **Accuracy:** The model achieves a high accuracy on the test dataset, indicating its effectiveness in detecting pneumonia.
- **Visualization:** The notebook provides visualizations of predictions, including a comparison of actual and predicted labels.

---

### **Contributing**

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or new features.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new pull request

---

### **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.






















