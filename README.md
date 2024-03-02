1.Model Architecture:
    *The CNN model is defined using Sequential from Keras.
    *It consists of three pairs of convolutional (Conv2D) and max-pooling (MaxPooling2D) layers followed by a flattening layer.
    *After the flattening layer, there's a dense layer with 512 units and ReLU activation, followed by a dropout layer with a dropout rate of 0.5 to reduce overfitting.
    *Finally, there's a dense layer with softmax activation, having the same number of units as the number of classes, which outputs the predicted class probabilities.

2.Model Compilation:
    *The model is compiled using the Adam optimizer and categorical cross-entropy loss function. Accuracy is chosen as the evaluation metric.

3.Data Augmentation:
    *An ImageDataGenerator is used for data augmentation, which rescales the image pixel values to the range [0, 1], applies random shear and zoom transformations, and flips images horizontally. This helps increase the diversity of the training data and improve the model's generalization.

4.Data Loading:
    *Training, validation, and test data are loaded using flow_from_directory from the ImageDataGenerator, specifying the directory paths, target size, batch size, and class mode.

5.Model Training:
    *The fit method is used to train the model on the training data, specifying the number of epochs and using the validation data for monitoring the model's performance during training.

6.Model Evaluation:
    *The trained model is evaluated on the test data using the evaluate method to compute the test loss and accuracy.

7.Per-Class Accuracy Calculation:
   *The script calculates the accuracy for each class (hand gesture) separately by comparing the predicted labels with the true labels in the test set.

8.Results Visualization:
   *The script plots the accuracy for each hand gesture as a bar chart and a pie chart to visualize the distribution of accuracies across different classes.

To use this script:

  *Replace the directory paths (train, validation, and test) with the paths to your own dataset folders containing images of hand gestures.
  *Ensure that your dataset is organized such that each class (hand gesture) has its own subdirectory within the train, validation, and test directories.
  *Execute the script using a Python interpreter.

This script demonstrates how to build and train a CNN model for image classification with TensorFlow and Keras, along with visualizing the results to analyze the model's performance on different classes.
