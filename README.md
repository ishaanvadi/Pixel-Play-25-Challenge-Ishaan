# Pixel-Play-25-Challenge-Ishaan
Pixel Play'25 Challenge - Animal Classification
Overview
In the Pixel Play'25 Challenge, the task was to classify images into 50 distinct animal classes. The training dataset consisted of images from 40 classes, and the test set contained images from all 50 classes, with 10 classes missing from the training data. I implemented a hybrid approach combining ****ResNet50**** for the 40 seen classes and **Zero-Shot Learning** for the 10 unseen classes.

**Approach**

**First, Classifying the 40 Seen Classes Using ResNet50**

**Then,Classifying the 10 Unseen Classes Using Zero-Shot Learning**

To classify the images from the 40 available classes, I used the ResNet50 architecture, pretrained on ImageNet. Here's a detailed explanation of the process:

ResNet50 Architecture:

I used the ResNet50 model without the top classification layer (i.e., include_top=False), which is common practice when utilizing a pretrained model for transfer learning. This allowed me to leverage the model's feature extraction capabilities.
I froze the weights of the initial 100 layers of ResNet50 to avoid overfitting and reduced the number of trainable parameters, while allowing the higher layers to be fine-tuned for my specific task.
Custom Classification Head:

After the ResNet50 base, I added a custom classification head:
GlobalAveragePooling2D layer to reduce the spatial dimensions.
Dense Layer with 1024 units and ReLU activation, followed by Dropout to prevent overfitting.
The final output layer has softmax activation with the number of units equal to the number of classes (num_classes = 40).
Optimizer and Regularization:

I used Adam optimizer with an initial learning rate of 1e-5 and L2 regularization on the weights to reduce overfitting.
The learning rate was dynamically adjusted using a LearningRateScheduler and ReduceLROnPlateau callbacks to reduce the learning rate when the validation loss plateaued.
Model Training:

The model was trained for 15 epochs with validation data, and I used callbacks to dynamically adjust the learning rate and monitor overfitting.

**2. Classifying the 10 Unseen Classes Using Zero-Shot Learning**

For the remaining 10 classes, which were not present in the training set, I used a Zero-Shot Learning approach to classify the images based on semantic relationships between classes.

**Low-Confidence Detection:**


During prediction on the test set, I monitored the confidence scores produced by the model for each image. If the confidence score for a predicted class was below a threshold of 0.04, the image was considered as a low-confidence prediction.
Low-confidence images were moved to a separate directory for further processing.

**Zero-Shot Learning:**

I applied Zero-Shot learning to the images that had low-confidence predictions. This method involved leveraging semantic relationships between classes, using predicates and the binary matrix for classification.
The classify_images function handles the loading and prediction for each image, and I used the softmax output of the model to classify the image with the highest probability.
For images with low confidence, I moved them into a separate folder (new_folder), allowing for further analysis and classification using a secondary algorithm or manual methods.

CSV Logging:

All predictions with confidence scores above the threshold (0.04) were logged into a CSV file (predict_with_confidence.csv), containing the image_id, predicted class, and confidence score.
The low-confidence images were separately logged in another CSV file (predict_above_threshold.csv) for analysis.

****3. Output Generation****

Final Predictions:
Predictions for the images with confidence scores ≥ 0.04 were saved in a CSV file for submission.
Low-confidence images were moved to a separate directory (test_new) for further processing.

****Results****

Accuracy for 40 Seen Classes: The model achieved high accuracy on the 40 available classes using ResNet50. The accuracy for the seen classes ranged from 93-95%.

Accuracy for 10 Unseen Classes: After applying the Zero-Shot learning mechanism to the low-confidence predictions, the model achieved an accuracy of 75-80% on the unseen classes.

Low-Confidence Detection: Images with confidence scores below 0.04 were successfully identified and moved to a new directory for further handling, ensuring that the predictions made on those images were treated cautiously.
****Challenges and Solutions****
Handling Unseen Classes:

The most significant challenge was handling the 10 unseen classes in the test set. The Zero-Shot learning technique provided a viable solution to this issue by allowing the model to classify images based on semantic class relationships.
Low-Confidence Predictions:

To ensure that the model's confidence was properly evaluated, I set a confidence threshold (0.04) for low-confidence predictions. This step helped in filtering out images that were difficult for the model to classify.

Model Overfitting:

To prevent overfitting, I used techniques such as Dropout (with a rate of 0.5) and L2 regularization on the dense layers. These helped ensure that the model generalized well on unseen data.

Data Augmentation:

To address any potential data imbalance or overfitting issues during training, I used random flipping and rotation to augment the dataset, making the model more robust to variations in the images.
**Future Work**

Enhanced Zero-Shot Learning:

I plan to explore more advanced techniques in Zero-Shot learning, such as utilizing semantic embeddings from larger pretrained models (e.g., CLIP or GPT-3) to improve accuracy on the unseen classes.

Alternative Architectures:

Experimenting with other models like EfficientNet or VGG16 could provide insights into whether a different architecture could improve performance or efficiency for both seen and unseen classes.

Class Imbalance Handling:

I plan to experiment with more advanced class balancing techniques, such as focal loss or oversampling/undersampling of minority classes.
Further Optimization:

Fine-tuning hyperparameters, such as learning rate schedules, dropout rates, and regularization coefficients, could further enhance the model’s performance.

****Conclusion****
By combining ResNet50 for the 40 seen classes and Zero-Shot learning for the 10 unseen classes, I was able to develop an effective animal image classification model for the Pixel Play'25 Challenge. This hybrid approach helped achieve high accuracy on both seen and unseen classes while maintaining efficiency and robustness.
