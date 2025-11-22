# Music Genre Classifier

A deep learning project that classifies music into 10 different genres using Convolutional Neural Networks (CNNs) and mel-spectrogram features. This project uses the GTZAN dataset to train a model that can identify genres such as blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Workflow Overview](#workflow-overview)
- [Getting Started](#getting-started)
- [File-by-File Explanation](#file-by-file-explanation)
- [Model Architecture](#model-architecture)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Training](#training)
- [Evaluation](#evaluation)
- [Dependencies](#dependencies)

## Project Overview

This project implements an end-to-end machine learning pipeline for music genre classification:

1. **Data Preparation**: Splits the GTZAN dataset into train/validation/test sets
2. **Feature Extraction**: Converts audio files to mel-spectrograms (2D representations of audio)
3. **Model Training**: Trains a CNN to classify genres from mel-spectrograms
4. **Model Evaluation**: Tests the trained model and generates classification reports

## Project Structure

```
music-genre-classifier/
├── data/
│   ├── dataset/          # Original GTZAN dataset (10 genres, 100 files each)
│   ├── train/            # Training split (70% of data)
│   ├── val/              # Validation split (15% of data)
│   └── test/             # Test split (15% of data)
├── src/
│   ├── utils.py          # Dataset splitting utility
│   ├── dataset.py        # PyTorch Dataset class for audio loading
│   ├── dataloader.py     # DataLoader creation functions
│   ├── model.py          # CNN model architecture
│   ├── train.py          # Training script
│   └── eval.py           # Evaluation script
├── model.pth             # Saved trained model weights
├── training_log.txt      # Training progress log
└── requirements.txt      # Python dependencies
```

## Workflow Overview

The workflow follows this sequence:

```
1. Data Preparation (utils.py)
   ↓
2. Dataset Loading (dataset.py + dataloader.py)
   ↓
3. Model Definition (model.py)
   ↓
4. Training (train.py)
   ↓
5. Evaluation (eval.py)
```

### Where Does the Workflow Start?

**The workflow starts with `src/utils.py`** - you must first split your dataset before training or evaluation.

1. **First Step**: Run `src/utils.py` to split the dataset
2. **Second Step**: Run `src/train.py` to train the model
3. **Third Step**: Run `src/eval.py` to evaluate the trained model

## Getting Started

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Ensure your GTZAN dataset is located in `data/dataset/` with the following structure:
```
data/dataset/
├── blues/
│   ├── blues.00000.wav
│   └── ...
├── classical/
├── country/
└── ... (10 genres total)
```

### 3. Split Dataset

Run the utility script to split the dataset into train/val/test sets:

```bash
python src/utils.py
```

This will create:
- `data/train/` - 70% of files (70 per genre)
- `data/val/` - 15% of files (15 per genre)
- `data/test/` - 15% of files (15 per genre)

### 4. Train the Model

```bash
python src/train.py
```

This will:
- Train the model for 40 epochs
- Save the best model (based on validation accuracy) to `model.pth`
- Log training progress to `training_log.txt`

### 5. Evaluate the Model

```bash
python src/eval.py
```

This will:
- Load the trained model from `model.pth`
- Evaluate on the test set
- Print accuracy and classification report

## File-by-File Explanation

### `src/utils.py` - Dataset Splitting Utility

**Purpose**: Splits the original GTZAN dataset into train/validation/test sets.

**Main Function**: `split_dataset()`

**What it does**:
- Takes the original dataset from `data/dataset/`
- Randomly shuffles files within each genre (with seed=42 for reproducibility)
- Splits files into 70% train, 15% validation, 15% test
- Copies files into organized folders: `data/train/`, `data/val/`, `data/test/`
- Creates genre subfolders in each split directory

**Parameters**:
- `source_dir`: Path to original dataset (`data/dataset/`)
- `output_dir`: Where to create split folders (`data/`)
- `train_ratio`: 0.7 (70%)
- `val_ratio`: 0.15 (15%)
- `test_ratio`: 0.15 (15%)
- `seed`: 42 (for reproducibility)

**When to run**: Once, before training. Only needed if you haven't split the dataset yet.

---

### `src/dataset.py` - PyTorch Dataset Class

**Purpose**: Defines how to load and preprocess audio files into mel-spectrograms.

**Main Class**: `GTZANDataset`

**What it does**:
1. **Initialization (`__init__`)**:
   - Scans the directory structure to find all `.wav` files
   - Maps genres to numeric labels (0-9)
   - Filters out corrupted files during initialization
   - Sets up mel-spectrogram transformation parameters

2. **Audio Loading (`__getitem__`)**:
   - Loads audio file using librosa
   - Resamples to 22050 Hz if needed
   - Pads or truncates to 30 seconds (660,000 samples)
   - Converts to mel-spectrogram (128 mel bins)
   - Normalizes the spectrogram
   - Applies data augmentation (time and frequency masking)
   - Returns mel-spectrogram tensor and label

**Key Parameters**:
- `sample_rate`: 22050 Hz (standard for audio ML)
- `n_mels`: 128 (number of frequency bins)
- `duration`: 30 seconds (fixed length for all audio)

**Output Shape**: `[1, 128, time_frames]` - A 2D image-like representation of audio

**Why mel-spectrograms?**
- Mel-spectrograms are 2D representations of audio (like images)
- Height = frequencies (mel bins)
- Width = time
- Pixel intensity = energy at that frequency/time
- CNNs excel at detecting patterns in these "images"

---

### `src/dataloader.py` - DataLoader Creation

**Purpose**: Creates PyTorch DataLoaders for train/val/test sets.

**Main Function**: `get_loaders()`

**What it does**:
- Creates three `GTZANDataset` instances (train, val, test)
- Wraps them in PyTorch `DataLoader` objects
- Sets batch size (default: 32)
- **Shuffles training data** (important for learning)
- **Does NOT shuffle validation/test** (for consistent evaluation)

**Parameters**:
- `batch_size`: 32 (number of samples processed together)

**Returns**: `(train_loader, val_loader, test_loader)`

**Why shuffle train but not val/test?**
- Training: Random order helps model learn better (prevents overfitting to order)
- Validation/Test: Consistent order makes evaluation reproducible

---

### `src/model.py` - CNN Architecture

**Purpose**: Defines the neural network architecture.

**Main Class**: `AudioNN`

**Architecture**:
```
Input: [batch, 1, 128, time_frames] (mel-spectrogram)
  ↓
Conv2d(1→32) + BatchNorm + ReLU + MaxPool
  ↓
Conv2d(32→64) + BatchNorm + ReLU + MaxPool
  ↓
Conv2d(64→128) + BatchNorm + ReLU + MaxPool
  ↓
Conv2d(128→256) + BatchNorm + ReLU + MaxPool
  ↓
Flatten
  ↓
Linear(dynamic→256) + ReLU + Dropout(0.3)
  ↓
Linear(256→10) (10 genres)
  ↓
Output: [batch, 10] (logits for each genre)
```

**Key Features**:
- **Dynamic FC layer**: `fc1` is created on first forward pass (handles variable input sizes)
- **Batch Normalization**: Stabilizes training
- **Dropout**: Prevents overfitting (0.3 = 30% neurons randomly disabled)
- **4 Convolutional Blocks**: Extract increasingly complex features
- **2 Fully Connected Layers**: Final classification

**Why dynamic fc1?**
- Different audio lengths produce different flattened sizes after convolutions
- The layer size is determined automatically on first forward pass

---

### `src/train.py` - Training Script

**Purpose**: Trains the model and saves the best version.

**Main Functions**:

1. **`train_epoch()`**:
   - Sets model to training mode
   - Iterates through training batches
   - Computes loss and backpropagates
   - Updates model weights
   - Returns average loss and accuracy for the epoch

2. **`validate()`**:
   - Sets model to evaluation mode (no gradient computation)
   - Evaluates on validation set
   - Returns validation loss and accuracy

3. **Main Training Loop**:
   - Creates model, optimizer (Adam), and loss function (CrossEntropyLoss)
   - Trains for 40 epochs
   - After each epoch:
     - Trains on training set
     - Validates on validation set
     - Saves model if validation accuracy improves
     - Logs progress to `training_log.txt`

**Configuration**:
- `num_epochs`: 40
- `batch_size`: 32
- `learning_rate`: 0.0003 (Adam optimizer)
- `weight_decay`: 1e-5 (L2 regularization)

**Outputs**:
- `model.pth`: Best model weights (saved when validation accuracy improves)
- `training_log.txt`: Detailed training history

**Training Strategy**:
- Uses validation set to prevent overfitting
- Only saves model when validation accuracy improves
- Uses early stopping implicitly (saves best model)

---

### `src/eval.py` - Evaluation Script

**Purpose**: Evaluates the trained model on the test set.

**Main Function**: `evaluate()`

**What it does**:
1. Loads the trained model from `model.pth`
2. Sets model to evaluation mode
3. Iterates through test set (no gradient computation)
4. Collects all predictions and true labels
5. Computes accuracy
6. Generates classification report (precision, recall, F1-score per genre)

**Model Loading**:
- Handles dynamic `fc1` layer by doing a dummy forward pass first
- Uses `strict=False` to handle architecture differences gracefully
- Falls back to untrained model if file not found

**Output**:
- Test accuracy (percentage)
- Classification report with per-genre metrics

**When to run**: After training is complete, to see final performance.

---

## Model Architecture Details

The model uses a **CNN architecture** because:

1. **Mel-spectrograms are 2D**: They can be treated as images
2. **CNNs detect patterns**: 
   - Horizontal lines → steady tones
   - Vertical lines → drum hits
   - Curved patterns → melodies
   - Stacked lines → chords/harmony
3. **Hierarchical features**: Early layers detect edges/rhythms, later layers detect complex musical patterns

**Architecture Flow**:
- **Convolutional Layers**: Extract spatial features from spectrograms
- **Pooling Layers**: Reduce dimensionality, make features translation-invariant
- **Fully Connected Layers**: Combine features for final classification

## Data Processing Pipeline

For each audio file:

1. **Load**: Read `.wav` file using librosa
2. **Resample**: Convert to 22050 Hz (if needed)
3. **Normalize Length**: Pad or truncate to 30 seconds
4. **Convert to Mel-Spectrogram**: 
   - Apply Short-Time Fourier Transform (STFT)
   - Convert to mel scale (human perception)
   - Result: 128 frequency bins × time frames
5. **Normalize**: Zero mean, unit variance
6. **Augment**: Apply time and frequency masking (training only)
7. **Return**: Tensor ready for model input

## Training

**Hyperparameters**:
- Optimizer: Adam (adaptive learning rate)
- Learning Rate: 0.0003
- Weight Decay: 1e-5 (L2 regularization)
- Batch Size: 32
- Epochs: 40
- Loss Function: CrossEntropyLoss

**Training Process**:
1. Model processes batches of mel-spectrograms
2. Computes predictions (logits for 10 genres)
3. Compares with true labels using CrossEntropyLoss
4. Backpropagates error to update weights
5. Repeats for all training samples (1 epoch)
6. Validates on validation set
7. Saves model if validation accuracy improves
8. Repeats for 40 epochs

**Monitoring**:
- Training loss/accuracy (should decrease/increase)
- Validation loss/accuracy (should track training, but not overfit)
- Best model saved automatically

## Evaluation

**Metrics**:
- **Accuracy**: Overall percentage of correct predictions
- **Per-Genre Metrics**:
  - Precision: Of predicted genre X, how many were actually X?
  - Recall: Of actual genre X, how many were predicted as X?
  - F1-Score: Harmonic mean of precision and recall

**Test Set**: 
- Held out during training (never seen by model)
- Represents real-world performance
- 15% of original dataset (15 files per genre)

## Dependencies

```
torch          # PyTorch deep learning framework
torchaudio     # Audio processing utilities
numpy          # Numerical computations
librosa        # Audio loading and processing
matplotlib     # Plotting (if needed for visualization)
tqdm           # Progress bars
scikit-learn   # Classification metrics
```

## Usage Summary

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Split dataset (first time only)
python src/utils.py

# 3. Train model
python src/train.py

# 4. Evaluate model
python src/eval.py
```

## Notes

- The model uses **dynamic layer sizing** for the first fully connected layer to handle variable input dimensions
- **Data augmentation** (time/frequency masking) is applied during training to improve generalization
- The **best model** (highest validation accuracy) is automatically saved
- Training progress is logged to `training_log.txt` for monitoring
- The dataset should be split **before** training or evaluation

## Troubleshooting

**Issue**: Model file not found during evaluation
- **Solution**: Ensure training completed successfully and `model.pth` exists

**Issue**: Corrupted audio files
- **Solution**: The dataset loader automatically skips corrupted files during initialization

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` in `train.py` and `eval.py`

**Issue**: Dataset not found
- **Solution**: Ensure GTZAN dataset is in `data/dataset/` with genre subfolders

---