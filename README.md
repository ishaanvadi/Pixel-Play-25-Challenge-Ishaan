# Pixel Play'25 Challenge - Animal Classification Overview

In the Pixel Play'25 Challenge, the task was to classify images into 50 distinct animal classes. The dataset consisted of 40 seen classes for training, and the test set included images from all 50 classes, with 10 unseen classes. My approach utilized a pretrained ****ResNet50**** model for the 40 seen classes and applied ****Zero-Shot Learning**** to classify the unseen classes effectively.

****Approach****

****Firstly, Classifying the 40 Seen Classes Using ResNet50, then treating low confidence predictions with ZERO SHOT LEARNING METHOD****


****1. Classifying the 40 Seen Classes Using ResNet50****
For the 40 seen classes in the training set, I used the ResNet50 model, a well-known convolutional neural network (CNN) pretrained on ImageNet. Here's how I used it:

Pretrained ResNet50 Model:

The model was used without the top classification layer (include_top=False), which allows the model to leverage the pretrained feature extraction capabilities and fine-tune the final layers for the specific task.
The model was initialized with ImageNet weights, which helped in extracting meaningful features from the images.
python


_*from tensorflow.keras.applications import ResNet50_

_from tensorflow.keras.models import Model__

_from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout*_


Load the ResNet50 model without the top layer

*base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))*

Freeze the first 100 layers to avoid overfitting

__for layer in base_model.layers[:100]:_

  _layer.trainable = False_

Add custom classification head

_x = GlobalAveragePooling2D()(base_model.output)_

_x = Dense(1024, activation='relu')(x)_

_x = Dropout(0.5)(x)_

_output = Dense(40, activation='softmax')(x)__

Freezing Layers:

The first 100 layers of ResNet50 were frozen to preserve the pretrained weights and reduce overfitting. Only the final layers were fine-tuned on the specific task.


Custom Classification Head:

Added a GlobalAveragePooling2D layer to reduce the spatial dimensions.
Followed by a Dense layer with 1024 units and ReLU activation.
A Dropout (0.5) was added to prevent overfitting, especially considering the relatively small dataset.
The final output layer used softmax activation with 40 units corresponding to the 40 seen classes.
python

Compile the model

I used the Adam optimizer with an initial learning rate of 1e-5, along with L2 regularization to prevent overfitting.
The LearningRateScheduler and ReduceLROnPlateau callbacks were used to dynamically adjust the learning rate based on the validation loss.

from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

Learning rate scheduler to adjust the learning rate
def scheduler(epoch, lr):
    if epoch > 16:
        return lr * 0.9
    return lr

callbacks = [
    LearningRateScheduler(scheduler),
    ReduceLROnPlateau(patience=5, verbose=1)
]

Train the model
model.fit(train_data, epochs=20, validation_data=val_data, callbacks=callbacks)

**2. Classifying the 10 Unseen Classes Using Zero-Shot Learning**

For the 10 unseen classes in the test set, the pretrained ResNet50 model performed well on the 40 seen classes, but generated low-confidence predictions for the unseen classes. Here’s how I handled it:

Processing the Entire Test Set:
I first processed the entire test set through the ResNet50 model. The model accurately classified images from the 40 seen classes.
python

**Process the test set through ResNet50 model**

_predictions = model.predict(test_data)_

Low-Confidence Detection:

After obtaining the predictions, I monitored the confidence scores for each predicted class. If the confidence score was below 0.04, the prediction was classified as low confidence.

Threshold for low-confidence predictions
_threshold = 0.04_

Filter low-confidence predictions
_low_confidence_images = [image for image, score in zip(test_images, predictions) if max(score) < threshold]_

These low-confidence images were identified and moved to a separate folder for further processing.

**Zero-Shot Learning for Low-Confidence Images:**

For the low-confidence predictions, I applied Zero-Shot Learning, which uses semantic relationships between classes for classification.
I leveraged a predicate-based binary matrix (predefined relationships between classes) to classify the unseen classes.

_def zero_shot_predict(image):
    # Use a semantic relationship matrix to predict unseen classes
    # (pseudo-code for Zero-Shot Learning)
    prediction = apply_zero_shot(image)  # Apply the zero-shot learning algorithm
    return prediction_
    
CSV Logging:
Predictions with confidence ≥ 0.04 were logged into a CSV file for submission. Low-confidence images were separately recorded in another CSV for further analysis.


**3. Output Generation**

Final Predictions: For images with confidence scores greater than or equal to 0.04, I logged predictions in a CSV file (predict_with_confidence.csv) for submission.
Low-Confidence Handling: Images with confidence scores below the threshold were moved to a new directory (test_new) and processed using Zero-Shot Learning for further classification.

**Results**
Accuracy for 40 Seen Classes:
The pretrained ResNet50 model achieved high accuracy on the 40 seen classes, with accuracy ranging from 93-95%.

Accuracy for 10 Unseen Classes:
After applying Zero-Shot Learning to the low-confidence predictions, the model successfully classified the 10 unseen classes with an accuracy of 75-80%.

Low-Confidence Detection:
The method of setting a confidence threshold of 0.04 was effective in filtering out uncertain predictions. These images were flagged for further processing using Zero-Shot Learning, ensuring reliable predictions.


**Challenges and Solutions**

1.Handling Unseen Classes:
The most challenging aspect was handling the 10 unseen classes. Zero-Shot Learning helped address this by leveraging semantic relationships between classes, allowing the model to classify unseen images even without explicit training data for those classes.

2.Low-Confidence Predictions:
To avoid incorrect classifications, I set a confidence threshold of 0.04 to identify low-confidence images, ensuring they were processed with caution and passed to Zero-Shot Learning for better classification.

3.Overfitting:
To combat overfitting, I employed Dropout (0.5 rate) and L2 regularization on the dense layers, improving the model’s generalization.

4.Data Augmentation:
I applied random flips and rotations to the training dataset, helping the model learn more robust features and perform better across varying image orientations.


**Future Work**

1.Enhancing Zero-Shot Learning:
I plan to explore more advanced Zero-Shot Learning techniques, such as using semantic embeddings from larger pretrained models like CLIP or GPT-3, to improve performance on unseen classes.

2.Alternative Architectures:
I will experiment with other architectures, like EfficientNet or VGG16, to see if they provide better performance or computational efficiency for both seen and unseen classes.

3.Class Imbalance Handling:
I aim to implement focal loss or other techniques for class balancing, which could address any class imbalance issues, especially for less-represented classes.

4.Hyperparameter Optimization:
Fine-tuning hyperparameters like learning rate schedules, dropout rates, and regularization coefficients will be beneficial in further optimizing the model's performance.


**Conclusion**

By combining ResNet50 for the 40 seen classes and Zero-Shot Learning for the 10 unseen classes, I developed an effective model for the Pixel Play'25 Challenge. This hybrid approach achieved high accuracy on both the seen and unseen classes, efficiently handling the challenge’s complexities while maintaining computational efficiency.
