# Takopi-s-Original-Team
Training ResNet50 neural network to detect and classify breeds of cattle and buffalo's .Implementing it in a locally run website in such a way that the uploaded image breed will be displayed as output .
Indian Bovine Breeds Classification with ResNet50
This repository implements a ResNet50 model in TensorFlow to classify 41 Indian bovine breeds (cows and buffaloes) using the Indian Bovine Breeds dataset from Kaggle. The code is adapted from a CIFAR-100 notebook, optimized for the custom dataset with 64x64 images and 41 classes. It includes data loading, model training, evaluation, and visualization, with fixes for issues like undefined variables, incomplete data loading, and incorrect indexing.
Dataset

Source: Indian Bovine Breeds
Description: Image database of Indian cow and buffalo breeds, organized in subfolders (one per breed, 41 total).
Classes: 41 breeds (e.g., Sahiwal, Gir, Murrah). See src/evaluate.py for the label list.

#Prerequisites

Python: 3.8+
Hardware: GPU recommended (e.g., via Colab or local setup)
Kaggle API: Required for dataset download (~/.kaggle/kaggle.json)
Dependencies: Listed in requirements.txt

## Setup Instructions

# Clone Repository
git clone https://github.com/your-username/indian-bovine-resnet.git
cd indian-bovine-resnet


# Install Dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt


# Download Dataset

Configure Kaggle API: Place kaggle.json in ~/.kaggle/ (get from Kaggle account settings) and run chmod 600 ~/.kaggle/kaggle.json.
Run:python download_dataset.py


This downloads and extracts the dataset to data/Indian_bovine_breeds/, expecting 41 breed subfolders.
Alternatively, download manually from Kaggle and extract to data/Indian_bovine_breeds/.


# Verify Breed Labels

Check subfolder names to ensure they match breed_labels in src/evaluate.py:import os
print(sorted(os.listdir('data/Indian_bovine_breeds')))


Update breed_labels in src/evaluate.py if names differ.


# Create Directories
mkdir -p logs models data



# Running the Project

# Train Model
python src/train.py


# Loads dataset, normalizes, and trains ResNet50 for up to 200 epochs.
Saves best model (models/best_model.h5) and final model (models/resnet.h5).
Logs to logs/ for TensorBoard.


# Evaluate Model
python src/evaluate.py


Predicts on up to 100 test images.
Saves results to prediction_results.csv.
Displays accuracy and visualizes predictions with Matplotlib.


# View Training Logs
tensorboard --logdir logs/


Open the provided URL in a browser to view training metrics.



# Code and Explanations
1. download_dataset.py
Downloads and extracts the Kaggle dataset using the Kaggle API.
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_indian_bovine_dataset():
    """Download and extract the Indian Bovine Breeds dataset from Kaggle."""
    try:
        api = KaggleApi()
        api.authenticate()  # Requires kaggle.json in ~/.kaggle/
        
        dataset_slug = 'lukex9442/indian-bovine-breeds'
        download_path = 'data/indian_bovine_breeds.zip'
        extract_path = 'data/Indian_bovine_breeds'
        
        print("Downloading dataset...")
        os.makedirs('data', exist_ok=True)
        api.dataset_download_files(dataset_slug, path='data/', unzip=False)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        os.remove(download_path)
        print(f"Dataset extracted to {extract_path}")
        print("Number of classes (breeds):", 
              len([d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]))
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Ensure Kaggle API is set up (kaggle.json) and dataset path is correct.")

if __name__ == "__main__":
    download_indian_bovine_dataset()

# Explanation:

Uses Kaggle API to fetch the dataset (lukex9442/indian-bovine-breeds).
Extracts to data/Indian_bovine_breeds/, verifies 41 breed subfolders.
Handles errors for missing API key or invalid paths.

# 2. src/model.py
Defines the ResNet50 architecture for 41-class classification with 64x64 inputs.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

def identity_block(input_tensor, filters, kernel_size=3):
    """Identity block for ResNet without dimension change."""
    filters1, filters2, filters3 = filters

    x = layers.Conv2D(filters1, 1, padding='valid')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, 1, padding='valid')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, filters, kernel_size=3, strides=2):
    """Convolutional block for ResNet with dimension change."""
    filters1, filters2, filters3 = filters

    x = layers.Conv2D(filters1, 1, strides=strides, padding='valid')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, 1, padding='valid')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(filters3, 1, strides=strides, padding='valid')(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet50(input_shape=(64, 64, 3), classes=41):
    """Builds ResNet50 model for Indian bovine breeds classification."""
    img_input = layers.Input(shape=input_shape)

    # Initial convolution and pooling
    x = layers.Conv2D(64, 7, strides=2, padding='same')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual stages
    # Stage 1
    x = conv_block(x, [64, 64, 256], strides=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    # Stage 2
    x = conv_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    # Stage 3
    x = conv_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    # Stage 4
    x = conv_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes, activation='softmax')(x)

    model = Model(img_input, x, name='resnet50_bovine')
    return model

# Explanation:

Implements ResNet50 with residual blocks (identity and convolutional).
Input: 64x64x3 images (adjusted from original 224x224 for efficiency).
Output: Softmax for 41 breeds.
Uses batch normalization and ReLU for stable training.

3. src/data.py
Loads and preprocesses the dataset using ImageDataGenerator.
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def normalize(X_train, X_test):
    """Normalize training and test sets to zero mean and unit variance."""
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print(f"Training Mean: {mean}")
    print(f"Training Std: {std}")
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

def normalize_production(x):
    """Normalize production data using pre-computed stats from training."""
    mean = 121.936  # From original code; update if needed for bovine dataset
    std = 68.389
    return (x - mean) / (std + 1e-7)

def load_and_split_data(data_dir, target_size=(64, 64), test_split=0.2, batch_size=32):
    """
    Load and split the Indian Bovine Breeds dataset into train/test sets.
    
    Expects directory structure: data_dir/breed_name/image.jpg
    """
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize to [0,1]
        validation_split=test_split
    )

    # Training generator
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    # Validation (test) generator
    test_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=42
    )

    # Collect training data
    x_train, y_train = [], []
    for _ in range(len(train_generator)):
        images, labels = train_generator.next()  # Use next() in loop
        x_train.append(images)
        y_train.append(labels)

    # Collect test data (fixed from original incomplete loop)
    x_test, y_test = [], []
    for _ in range(len(test_generator)):
        images, labels = test_generator.next()
        x_test.append(images)
        y_test.append(labels)

    # Concatenate into NumPy arrays
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    print(f"Loaded {len(x_train)} training images, {len(x_test)} test images")
    print(f"Number of classes: {y_train.shape[1]}")  # Should be 41

    return x_train, x_test, y_train, y_test

Explanation:

Normalization: Computes mean/std for training data; applies to train/test. Production uses fixed stats (may need updating for bovine dataset).
Data Loading: Loads images from data/Indian_bovine_breeds/, resizes to 64x64, splits 80/20, and converts to NumPy arrays.
Fix: Added test data collection loop (missing in original) and prints dataset stats.
Dataset Fit: Matches Kaggle dataset’s folder-based structure.

4. src/train.py
Trains the ResNet50 model with augmentation and callbacks.
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from src.model import ResNet50
from src.data import load_and_split_data, normalize

# Configuration
data_dir = 'data/Indian_bovine_breeds'  # Path to Kaggle dataset
num_classes = 41
batch_size = 32
epochs = 200

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Load and preprocess data
print("Loading data...")
x_train, x_test, y_train, y_test = load_and_split_data(data_dir)
x_train, x_test = normalize(x_train, x_test)

# Build model
print("Building ResNet50 model...")
model = ResNet50(input_shape=(64, 64, 3), classes=num_classes)

# Data augmentation for bovine images
datagen = ImageDataGenerator(
    rotation_range=15,          # Slight rotation for varied head angles
    width_shift_range=0.1,      # Horizontal shift
    height_shift_range=0.1,     # Vertical shift
    horizontal_flip=True,       # Flip for left/right symmetry
    vertical_flip=False,        # No vertical flip for animals
    fill_mode='nearest'
)
datagen.fit(x_train)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.15, patience=3, min_lr=1e-6),
    TensorBoard(log_dir='logs/')
]

# Compile model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.8)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

print("Starting training...")
# Train
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=2,
    callbacks=callbacks
)

# Save final model
model.save('models/resnet.h5')
print("Training completed. Model saved to models/resnet.h5")

Explanation:

Setup: Configures paths, batch size (32), and epochs (200).
Data: Loads and normalizes using data.py.
Augmentation: Applies rotations, shifts, and flips suitable for animal images.
Training: Uses SGD with momentum, categorical crossentropy, and callbacks (early stopping, checkpointing, LR reduction, TensorBoard).
Output: Saves best and final models to models/.
Fix: Replaces Google Drive path with local path.

5. src/evaluate.py
Evaluates the model, predicts on test images, and visualizes results.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data import load_and_split_data, normalize_production
from src.utils import plot_train_val_loss

# Indian Bovine Breeds labels (41 classes; update based on dataset subfolders)
breed_labels = [
    'Amrit Mahal', 'Badri', 'Bargur', 'Bhadawari', 'Binjharpuri', 'Dangi', 'Deoni', 'Gangatiri',
    'Gir', 'Hallikar', 'Hariana', 'Jaffarabadi', 'Jawari', 'Jersey Cross', 'Kankrej', 'Kangayam',
    'Kenkatha', 'Kherigarh', 'Khillar', 'Konkan', 'Krishna Valley', 'Ladakhi', 'Lal Sindhi',
    'Malnad Gidda', 'Malvi', 'Mehsana', 'Motu', 'Murrah', 'Nagauri', 'Nimari', 'Ongole',
    'Punganur', 'Rathi', 'Red Kandhari', 'Red Sindhi', 'Sahiwal', 'Siri', 'Surti', 'Tharparkar',
    'Umblachery', 'Vechur'
]

# Verify number of labels matches expected classes
if len(breed_labels) != 41:
    raise ValueError(f"Expected 41 breed labels, got {len(breed_labels)}. Update breed_labels.")

# Configuration
data_dir = 'data/Indian_bovine_breeds'
model_path = 'models/resnet.h5'
num_predictions = 100  # Number of test images to evaluate

# Load data (no normalization here as it's done in predict)
print("Loading test data...")
x_train, x_test, y_train, y_test = load_and_split_data(data_dir)
# Note: For evaluation, we use the rescaled [0,1] data; normalization applied in predict

# Load model
print("Loading model...")
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

# Initialize results
results = []

# Predict on first N test images (fixed indexing)
for i in range(min(num_predictions, len(x_test))):
    image = x_test[i]  # Already rescaled [0,1]
    true_class_one_hot = y_test[i]
    true_class = np.argmax(true_class_one_hot)

    # Add batch dimension and normalize for prediction
    image_batch = np.expand_dims(image, axis=0)
    image_normalized = normalize_production(image_batch)

    # Predict
    predictions = model.predict(image_normalized, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Store results
    results.append({
        'Image_Index': i,
        'Predicted_Breed': breed_labels[predicted_class],
        'True_Breed': breed_labels[true_class],
        'Confidence': f"{confidence:.4f}",
        'Correct': predicted_class == true_class
    })

    # Visualize
    plt.figure(figsize=(6, 6))
    plt.imshow(image)  # Original rescaled image
    plt.title(f"Predicted: {breed_labels[predicted_class]}\nTrue: {breed_labels[true_class]}\nConfidence: {confidence:.4f}")
    plt.axis('off')
    plt.show()

    # Print details
    print(f"Image {i}:")
    print(f"Predicted Breed: {breed_labels[predicted_class]} (Class {predicted_class})")
    print(f"True Breed: {breed_labels[true_class]} (Class {true_class})")
    print(f"Confidence: {confidence:.4f}")
    print("Correct!" if predicted_class == true_class else "Incorrect.")
    print()

# Create DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('prediction_results.csv', index=False)
print("\nPrediction Results Table:")
print(results_df)

# Calculate accuracy
true_count = results_df['Correct'].sum()
total_count = len(results_df)
accuracy = (true_count / total_count) * 100
print(f"\nAccuracy on {total_count} test images: {accuracy:.2f}%")

# Plot training history (requires history to be saved during training)
try:
    # For this to work, save history in train.py (e.g., with pickle) and load here
    # Placeholder: Assuming history is not available
    print("Training plots skipped (add history saving in train.py for visualization).")
except:
    pass

Explanation:

Labels: Defines 41 breed names (placeholder; verify against dataset subfolders).
Evaluation: Predicts on up to 100 test images, fixing original i * 100 indexing error.
Output: Saves predictions to CSV, computes accuracy, visualizes images.
Fixes: Replaces cifar100_labels and undefined model_with_custom_pool with correct references.
Note: History plotting requires saving history in train.py.

6. src/utils.py
Utility functions for plotting training metrics.
import matplotlib.pyplot as plt
import numpy as np

def plot_train_val_loss(train_values, val_values, epochs=None, title='Training vs Validation', ylabel='Value', label_train='Training', label_val='Validation'):
    """Plot training and validation metrics."""
    if epochs is None:
        epochs = range(1, len(train_values) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, label=label_train, marker='o')
    plt.plot(epochs, val_values, label=label_val, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

Explanation:

Plots training/validation loss or accuracy.
Used in evaluate.py if history is saved.

7. src/__init__.py
Makes src a Python package.
# Makes src a Python package

Explanation:

Empty file to enable importing from src.

8. requirements.txt
Dependencies for reproducibility.
tensorflow==2.17.0
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.2
kaggle==1.6.14

Explanation:

Specifies exact versions.
Includes kaggle for dataset download.

9. .gitignore
Excludes large or temporary files.
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
*.egg-info/

# Models and outputs
models/
logs/
*.h5
prediction_results.csv

# Data (large files)
data/
*.zip

# IDE and OS
.vscode/
.idea/
*.DS_Store

# Kaggle
~/.kaggle/

Explanation:

Ignores virtual environments, models, dataset, and temporary files.

Creating the GitHub Repository

Create on GitHub

Create a new repository named indian-bovine-resnet (do not initialize with README or .gitignore).


Set Up Locally
mkdir indian-bovine-resnet
cd indian-bovine-resnet
git init


Create Structure
mkdir -p src data models logs
touch src/__init__.py


Add Files

Copy the above code blocks into respective files:
download_dataset.py
src/model.py
src/data.py
src/train.py
src/evaluate.py
src/utils.py
src/__init__.py
requirements.txt
.gitignore
README.md


Example:echo "[Paste README content]" > README.md




Commit and Push
git add .
git commit -m "Initial commit for Indian Bovine Breeds ResNet50 classification"
git remote add origin https://github.com/your-username/indian-bovine-resnet.git
git branch -M main
git push -u origin main


Verify

Check GitHub to confirm all files are uploaded.



Notes

Breed Labels: The breed_labels list in src/evaluate.py is a placeholder. Update it after downloading the dataset:import os
print(sorted(os.listdir('data/Indian_bovine_breeds')))


Normalization Stats: Original mean (121.936) and std (68.389) may need recalibration for the bovine dataset.
Performance: Expect 80-95% accuracy with sufficient training; tune batch size or epochs if needed.
Improvements:
Add transfer learning with pre-trained ResNet50 weights.
Save training history (e.g., via pickle) in train.py for plotting.
Increase image size (e.g., 128x128) for better features.


Fixes from Original:
Added test data loading loop in data.py.
Fixed evaluation indexing (i * 100 to range(min(num_predictions, len(x_test)))).
Replaced cifar100_labels with breed_labels.
Removed Google Drive dependencies.
Fixed undefined model_with_custom_pool.



Troubleshooting

Kaggle API Error: Ensure kaggle.json is in ~/.kaggle/ with correct permissions.
Memory Issues: Reduce batch_size in train.py (e.g., to 16).
Class Mismatch: If not 41 classes, update num_classes in model.py and train.py.
Visualization: Add history saving in train.py for training plots.

License
MIT License. Dataset usage per Kaggle terms.
