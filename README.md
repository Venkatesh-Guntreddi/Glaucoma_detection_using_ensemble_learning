# glaucoma_detection_using_ensemble_learning

## Project Overview

This project focuses on developing and evaluating deep learning models for the automated detection of Glaucoma from retinal fundus images. Glaucoma is a leading cause of irreversible blindness worldwide, and early detection is crucial for effective treatment and preserving vision. This repository implements and compares three popular convolutional neural network (CNN) architectures (ResNet50, VGG16) and a Vision Transformer (Swin Transformer), culminating in an ensemble model for improved performance.

## Dataset

The project utilizes the **ORIGA (Online Retinal Fundus Image Database for Glaucoma Analysis)** dataset, specifically a subset provided as `Glaucoma.zip` and `glaucoma.csv`. Due to its size, the dataset files are not directly included in this repository.

**Download Instructions:**
To run this project, you need to download the dataset and place it in the correct directory.
1.  Download `Glaucoma.zip` and `glaucoma.csv` from [**Dataset Download Link Here**](https://www.kaggle.com/datasets/venkatesh0003/glaucoma).
2.  Place `glaucoma.csv` directly in the root of your cloned project directory (e.g., `/content/glaucoma.csv` if running in Colab, or `./glaucoma.csv` locally).
3.  Unzip `Glaucoma.zip` into a directory structure such that the images are accessible at `[YOUR_PROJECT_ROOT]/ORIGA/ORIGA/Images`. For example, if you unzip it, you might get a folder named `ORIGA` containing another `ORIGA` folder inside, and then an `Images` folder. Ensure this path aligns with `"/content/ORIGA/ORIGA/Images"` or your adjusted local path.

* **Images**: Fundus images stored in `.jpg` format.
* **CSV**: Contains metadata, including filenames and Glaucoma diagnosis labels (0: No Glaucoma, 1: Glaucoma).

## Methodology

The project pipeline involves the following key steps:

1.  **Data Loading and Preprocessing**:
    * Loading retinal fundus images and associated diagnostic labels from a CSV file.
    * Initial data exploration to understand class distribution and identify missing values.
2.  **Data Organization**:
    * Images are organized into 'yes' (Glaucoma) and 'no' (No Glaucoma) categories.
3.  **Data Augmentation**:
    * To address class imbalance (if any) and improve model generalization, the 'Glaucoma' class images are augmented using various transformations (rotation, shifts, shear, zoom, horizontal flip). This significantly increases the number of samples for the minority class.
4.  **Data Splitting**:
    * The combined dataset (original 'no' and augmented 'yes' images) is split into training (70%), validation (15%), and test (15%) sets, ensuring stratification to maintain class balance across splits.
5.  **Model Architectures**:
    * **ResNet50**: A widely used residual network known for its deep architecture and ability to learn complex features. Pre-trained on ImageNet and fine-tuned for our task.
    * **VGG16**: A deep convolutional network with a simple, uniform architecture, also pre-trained on ImageNet and fine-tuned.
    * **Swin Transformer**: A powerful Vision Transformer model that adapts the Transformer architecture for computer vision tasks by leveraging shifted windows. Pre-trained and fine-tuned.
6.  **Ensemble Modeling**:
    * An ensemble approach is implemented by averaging the predicted probabilities from ResNet50, VGG16, and Swin Transformer models. This aims to leverage the strengths of individual models and achieve more robust and accurate predictions.
7.  **Training and Evaluation**:
    * Models are trained using appropriate loss functions (Categorical Crossentropy for Keras, CrossEntropyLoss for PyTorch) and optimizers (Adam).
    * Learning rate scheduling and early stopping are employed to optimize the training process.
    * Performance is evaluated using metrics such as:
        * Accuracy
        * Classification Report (Precision, Recall, F1-score)
        * Confusion Matrix
        * Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)

## Model Saving

The trained ResNet50, VGG16, and Swin Transformer models (or their weights) are saved to the `/content/` directory (or specified save path) after training.

* `resnet50_model.h5`
* `resnet50_model_weights.weights.h5`
* `vgg16_model.h5`
* `vgg16_model_weights.weights.h5`
* `best_swin_model.pth`

These can be loaded for future inference without retraining.

## Future Work

* Explore other advanced CNN architectures (e.g., EfficientNet, DenseNet).
* Investigate more sophisticated ensemble techniques (e.g., stacking, boosting).
* Integrate attention mechanisms for better feature focusing.
* Test on a larger and more diverse dataset for better generalization.
* Develop a user-friendly interface for easier inference.

## License

Copyright (c) 2025 [Guntreddi Venkatesh]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgements

* ORIGA Dataset
* TensorFlow/Keras, PyTorch, Timm libraries
* Researchers whose work inspired this project.
