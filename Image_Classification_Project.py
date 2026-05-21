# ============================================================================
# IMAGE CLASSIFICATION MODEL - Complete Machine Learning Project
# ============================================================================
# This project demonstrates a complete image classification pipeline using:
# - CNN (Convolutional Neural Network) with Keras/TensorFlow
# - CIFAR-10 Dataset (10 classes of objects)
# - Model training, evaluation, and prediction
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================================================================
# PART 1: LOAD AND EXPLORE DATA
# ============================================================================

print("=" * 70)
print("STEP 1: LOADING CIFAR-10 DATASET")
print("=" * 70)

# Load the CIFAR-10 dataset (60,000 images, 32x32 pixels, 3 color channels)
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Class names in CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training data shape: {x_train.shape}")  # (50000, 32, 32, 3)
print(f"Testing data shape: {x_test.shape}")    # (10000, 32, 32, 3)
print(f"Number of classes: {len(class_names)}")
print(f"Image dimensions: 32x32 pixels with 3 color channels (RGB)")

# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: DATA PREPROCESSING")
print("=" * 70)

# Normalize pixel values to 0-1 range (instead of 0-255)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(" Normalized pixel values to [0, 1] range")

# Convert class labels to one-hot encoding
# Example: class 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(" Converted labels to one-hot encoding")
print(f"  Example: Class label shape is now {y_train[0].shape}")

# ============================================================================
# PART 3: BUILD CNN MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: BUILDING CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("=" * 70)

model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 output classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Architecture:")
model.summary()

# ============================================================================
# PART 4: TRAIN THE MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: TRAINING THE MODEL")
print("=" * 70)

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1
)

print("\n Model training completed!")

# ============================================================================
# PART 5: EVALUATE MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: MODEL EVALUATION")
print("=" * 70)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Get predictions
y_pred_prob = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred, target_names=class_names))

# ============================================================================
# PART 6: VISUALIZE RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: VISUALIZATION")
print("=" * 70)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/training_history.png', dpi=300, bbox_inches='tight')
print(" Saved: training_history.png")
plt.close()

# ============================================================================
# PART 7: SAMPLE PREDICTIONS
# ============================================================================

print("\nSample Predictions on Test Set:")
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for i in range(15):
    ax = axes[i // 5, i % 5]
    
    # Get image
    img = x_test[i]
    
    # Get prediction
    pred_class = y_pred[i]
    true_class = y_test_labels[i]
    confidence = y_pred_prob[i][pred_class]
    
    # Plot
    ax.imshow(img)
    
    # Color: green if correct, red if wrong
    color = 'green' if pred_class == true_class else 'red'
    
    ax.set_title(
        f"True: {class_names[true_class]}\n"
        f"Pred: {class_names[pred_class]}\n"
        f"Conf: {confidence:.2f}",
        color=color,
        fontsize=9,
        fontweight='bold'
    )
    ax.axis('off')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sample_predictions.png', dpi=300, bbox_inches='tight')
print(" Saved: sample_predictions.png")
plt.close()

# ============================================================================
# PART 8: CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(y_test_labels, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Number of Samples'})
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix - CIFAR-10 Image Classification', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix.png")
plt.close()

# ============================================================================
# PART 9: MAKE PREDICTIONS ON CUSTOM IMAGES
# ============================================================================

print("\n" + "=" * 70)
print("STEP 7: FUNCTION FOR CUSTOM PREDICTIONS")
print("=" * 70)

def predict_custom_image(image_array):
    """
    Predict the class of a custom image.
    
    Parameters:
    -----------
    image_array : numpy array of shape (32, 32, 3) with values in [0, 1]
    
    Returns:
    --------
    predicted_class : string (class name)
    confidence : float (probability of prediction)
    all_probabilities : dict (all class probabilities)
    """
    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Get prediction
    prediction = model.predict(image_batch, verbose=0)
    predicted_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_idx]
    
    # Create probability dictionary
    probs = {class_names[i]: float(prediction[0][i]) for i in range(10)}
    
    return class_names[predicted_idx], confidence, probs

# Example: Predict on first test image
test_image = x_test[0]
pred_class, confidence, all_probs = predict_custom_image(test_image)

print(f"\nExample Prediction:")
print(f"Predicted Class: {pred_class}")
print(f"Confidence: {confidence:.4f}")
print(f"\nAll Class Probabilities:")
for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cls:15s}: {prob:.4f}")

print("\n" + "=" * 70)
print(" PROJECT COMPLETE!")
print("=" * 70)
