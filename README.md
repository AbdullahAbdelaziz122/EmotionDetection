# Facial Emotion Recognition using CNN

**Project Overview**

This project aims to classify facial emotions into seven categories (angry, disgusted, fearful, happy, sad, surprised, neutral) using a Convolutional Neural Network (CNN). The model was trained and evaluated on the FER2013 dataset.

**Dataset**

The project utilized the publicly available FER2013 dataset, consisting of 48x48 pixel grayscale images of faces with associated emotion labels.

**Model Architecture**

A Convolutional Neural Network (CNN) was employed for this task. The model architecture is as follows:

* Three convolutional layers with 32, 64, and 128 filters respectively, each followed by batch normalization and max pooling.
* Dropout layers with rates of 0.25 after each convolutional block.
* A flatten layer to convert the 2D feature maps to a 1D vector.
* Two dense layers with 128 and 7 neurons, using ReLU and softmax activation functions respectively.

**Data Preprocessing**

Before training the model, the FER2013 dataset underwent several preprocessing steps to ensure optimal performance:

1. **Image Resizing**: The original 48x48 pixel grayscale images were resized to a standardized input size suitable for the CNN architecture. This resizing step helps maintain consistency across all images.

2. **Normalization**: The pixel values of the resized images were normalized to a range between 0 and 1. This normalization step helps the model converge faster during training and improves overall performance.


3. **Label Encoding**: The emotion labels associated with each image were encoded into numerical values. This step is necessary for the model to understand and predict the emotion categories accurately.

By performing these preprocessing steps, we ensure that the input data is in a suitable format for training the CNN model and that it captures the necessary variations and patterns present in the FER2013 dataset.


**Training Process**
The model was trained using the Adam optimizer and sparse categorical crossentropy loss. The batch size was [insert batch size], and the model was trained for [insert number] epochs.

**Results**
* **Training Accuracy:** 80.92%
* **Validation Accuracy:** 80.92%
* **Test Accuracy:** 80.60%


### Interpretation

* **Good Generalization:** The close similarity between training, validation, and test accuracy suggests that the model is generalizing well to unseen data. Overfitting is not a major concern in this case.
* **Room for Improvement:** While the accuracy is decent, there's still potential for improvement.

### Potential Next Steps

1. **Data Augmentation:** this technique can help improve model robustness and accuracy.
2. **Hyperparameter Tuning:** Experiment with different hyperparameters to optimize performance.
3. **Model Architecture:** Consider exploring more complex architectures or transfer learning.
4. **Ensemble Methods:** Combining multiple models can sometimes boost performance.


### Additional Considerations

* **Error Analysis:** Analyze the types of errors the model makes to identify areas for improvement.
* **Visualization:** Visualize model predictions and activations to gain insights into its behavior.

**Given the relatively good performance and the potential for improvement, exploring data augmentation and hyperparameter tuning seems like promising next steps.**



**Dependencies**
* Python
* NumPy
* OpenCV
* TensorFlow/Keras