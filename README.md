****Pixel Play'25 Challenge - Animal Classification Overview****
In the Pixel Play'25 Challenge, the task was to classify images into 50 distinct animal classes. The training dataset consisted of images from 40 classes, and the test set contained images from all 50 classes, including 10 unseen classes. My approach leveraged a pretrained ResNet50 model for the 40 seen classes and Zero-Shot Learning to handle the unseen classes.

****Approach****

****Firstly, Classifying the 40 Seen Classes Using ResNet50, then treating low confidence predictions with  **ZERO SHOT LEARNING METHOD******


For the 40 classes that were available in the training set, I used ResNet50, a pretrained convolutional neural network (CNN) model known for its strong feature extraction capabilities. Here's how I utilized it:


Pretrained ResNet50 Model:


I used the ResNet50 model without the top classification layer (include_top=False) to benefit from the pretrained feature extraction while adding a custom classification head for my specific task.
The model was pretrained on ImageNet, and I fine-tuned the last layers to adapt to my specific dataset.

Freezing Layers:


I froze the first 100 layers of the ResNet50 base to prevent overfitting and reduce the number of trainable parameters, while allowing the higher layers to be fine-tuned for my specific dataset.

Custom Classification Head:

After the ResNet50 base, I added a GlobalAveragePooling2D layer, followed by a Dense layer (1024 units with ReLU activation) and Dropout (0.5 rate) to mitigate overfitting.
The final output layer used softmax activation with the number of units equal to 40 (for the 40 seen classes).

Optimizer and Regularization:


I used the Adam optimizer with an initial learning rate of 1e-5 and added L2 regularization to the weights.
A LearningRateScheduler and ReduceLROnPlateau callbacks dynamically adjusted the learning rate based on the validation loss during training.

Training:


The model was trained for 20 epochs, with validation data included to monitor overfitting. The learning rate adjustments helped improve model performance during training.

****2. Classifying the 10 Unseen Classes Using Zero-Shot Learning****

In the test set, there were images from the 10 unseen classes. The pretrained ResNet50 model performed well on the 40 seen classes but produced low-confidence predictions on the images belonging to these unseen classes. To handle this:

****Processing the Entire Test Set:****

First, I put the entire test set through the ResNet50 model. This allowed the model to classify the images from the 40 seen classes accurately.

****Low-Confidence Detection:****

After processing the test set, I monitored the confidence scores for each image. If the confidence score for a predicted class was below a threshold of 0.04, the image was considered as a low-confidence prediction.
These low-confidence images, which were likely from the unseen classes, were then moved to a separate directory for further processing.

****Zero-Shot Learning:****

For the images with low-confidence predictions, I applied a Zero-Shot Learning approach. This technique leverages semantic class relationships to predict classes for which the model has not been explicitly trained.

I used predicates and a binary matrix (predefined relationships between classes) for classifying these low-confidence images. The images were processed and classified based on these relationships, and the Zero-Shot Learning method provided the correct predictions for these unseen classes.

CSV Logging:

Predictions with a confidence score greater than 0.04 were logged into a CSV file (predict_with_confidence.csv), containing the image_id, predicted class, and confidence_score.
Low-confidence images were separately recorded in a second CSV file (predict_above_threshold.csv) for further analysis and confirmation.

****3. Output Generation****

Final Predictions:

For images with confidence scores greater than or equal to 0.04, I logged predictions in the CSV file (predict_with_confidence.csv) for submission.

Low-Confidence Handling:
Low-confidence images were moved to a new directory (test_new) and handled through Zero-Shot Learning for further analysis and accurate classification.

****Results****

Accuracy for 40 Seen Classes: The pretrained ResNet50 model achieved high accuracy on the 40 seen classes, with an accuracy range of 93-95% on these images.

Accuracy for 10 Unseen Classes: After applying Zero-Shot Learning to the low-confidence predictions, the model successfully classified the unseen classes with an accuracy of 75-80%.

Low-Confidence Detection: The approach of detecting low-confidence predictions (using a threshold of 0.04) was effective in filtering out uncertain predictions. These images were isolated for further processing using Zero-Shot Learning, ensuring the model's confidence was properly evaluated.

****Challenges and Solutions****

Handling Unseen Classes:

The most challenging aspect was handling the 10 unseen classes. The Zero-Shot Learning technique provided a solution by leveraging semantic class relationships, allowing the model to classify images even without explicit training data for those classes.

Low-Confidence Predictions:

To prevent incorrect predictions, I set a confidence threshold of 0.04 to identify low-confidence images. This step ensured that images the model was unsure about were processed with caution and passed to Zero-Shot Learning for better classification.

Overfitting:

To avoid overfitting, I employed Dropout (0.5 rate) and L2 regularization on the dense layers of the model. These techniques helped improve the model’s generalization, especially when training on the 40 seen classes.

Data Augmentation:

I used random flipping and rotation to augment the training dataset. These augmentations helped improve model robustness by simulating different orientations and perspectives of the animals in the images.

****Future Work****
Enhancing Zero-Shot Learning:

I plan to explore more advanced Zero-Shot learning techniques, such as using semantic embeddings from larger pretrained models (e.g., CLIP, GPT-3) for better performance on unseen classes.

Alternative Architectures:

Trying out different architectures, such as EfficientNet or VGG16, might yield better results or offer more computational efficiency for both seen and unseen classes.

Class Imbalance Handling:

Implementing more advanced class balancing techniques, like focal loss or oversampling/undersampling, could help address any class imbalance issues, particularly for classes with fewer examples.

Hyperparameter Optimization:

Fine-tuning hyperparameters such as learning rate schedules, dropout rates, and regularization coefficients could further improve the model’s performance on both seen and unseen classes.

****Conclusion****

By using ResNet50 for the 40 seen classes and Zero-Shot Learning for the 10 unseen classes, I developed an effective animal classification model for the Pixel Play'25 Challenge. The hybrid approach successfully handled both seen and unseen classes, achieving high accuracy while maintaining computational efficiency. This approach not only leveraged transfer learning but also creatively utilized Zero-Shot Learning for more robust and accurate predictions on the unseen data.

