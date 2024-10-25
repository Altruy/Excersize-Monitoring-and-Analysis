# Pose-Based Exercise Detection and Analysis

This project enables real-time exercise pose classification and analysis through `main.py`, which serves as the central script to run pose detection, classify exercises, count repetitions, and provide corrective feedback. Built with [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose) and PyTorch, the project supports both traditional landmark-based classification and neural network-based pose classification.

## Key Features
- **Exercise Pose Detection**: Identifies and classifies various exercise poses (e.g., push-ups, squats) from video input or webcam.
- **Neural Network-Based Classification**: Supports a neural network model for classifying poses with higher accuracy.
- **Repetition Counting**: Tracks repetitions based on pose transitions.
- **Real-Time Feedback**: Provides corrective feedback for each detected pose, assisting users in maintaining proper form.

## Directory Structure
- **main.py**: Primary script for executing training, testing, or live classification.
- **PoseClass.py**: Provides core utilities for pose detection, classification, and repetition counting.
- **NeuralNetwork.py**: Defines and trains a neural network for pose classification.
- **Dataloader.py**: Loads and prepares pose data for training/testing.
- **data_prep.py**: Downloads and processes video data into frames for training.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pose-exercise-analysis.git
   cd pose-exercise-analysis
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started with main.py

The `main.py` script offers multiple functionalities: neural network training, model evaluation, and real-time exercise classification. The default setup performs exercise classification on a sample video using a pre-trained neural network.

### Running Pose Detection and Classification
To use `main.py` for real-time exercise classification:
```bash
python main.py
```
By default, it will run the `exercize_NN` function, which uses a trained neural network model to detect poses and count repetitions. The output will be saved to `demo/output_nn.mp4`.

### main.py Structure and Usage

`main.py` is structured around the following core functions:

1. **`exercize`**:
   - Uses the `PoseClass` from `PoseClass.py` to detect and classify poses directly using MediaPipe.
   - Counts repetitions and overlays pose labels and rep counts on the video output.

2. **`ClassifyPose_NN`**:
   - A helper function to classify poses using the neural network model.
   - Takes in detected landmarks and returns a predicted pose label.

3. **`exercize_NN`**:
   - Utilizes `ClassifyPose_NN` to classify poses from a video input using a trained neural network.
   - Displays classified poses with labels and repetition counts in real time, and saves the output to `output_nn.mp4`.

4. **`train_NN`**:
   - Trains the neural network model on pre-extracted landmark data.
   - Takes a CSV file with labeled landmark data, trains the model, and saves the trained weights.
   - To run training, uncomment the `train_NN` section in `main.py` and adjust `epochs`, `data_file`, and `weights`.

5. **`test_NN`**:
   - Evaluates the trained model on a test dataset to measure accuracy.
   - Displays a confusion matrix to show the model’s performance across different exercise classes.
   - To run testing, uncomment `test_NN` in `main.py` and specify the model weights file.

### Sample Configurations in main.py

The `__main__` section of `main.py` includes configurations for:
- **Landmark Mapping**: Identifies the body joints to track during exercises (e.g., shoulders, hips, knees).
- **Label Mapping**: Maps exercise labels to integer values for classification.
- **Training and Testing**: Uncomment relevant sections to train or test the neural network model.

#### Example: Training the Neural Network
1. Set parameters in the `train_NN` function, such as:
   ```python
   data_file = './dataset.csv'
   epochs = 3000
   weights = None
   prev_epoch = 0
   ```
2. Uncomment `train_NN` in `main.py`:
   ```python
   nn, weights = train_NN(dataset, landmarks_tracked, labels_tracked, landmark_labels, epochs, weights=weights, prev_epoch=prev_epoch)
   ```

3. Run:
   ```bash
   python main.py
   ```
   The model will save the trained weights and display a loss plot.

#### Example: Testing the Neural Network
To evaluate a trained model, set the `weights` variable to the saved model weights file and uncomment the `test_NN` section in `main.py`:
```python
test_NN(dataset, weights, landmarks_tracked, labels_tracked, landmark_labels)
```
Run `python main.py` to display the model’s accuracy and confusion matrix.

## Additional Scripts Overview

- **PoseClass.py**: Contains `PoseClass` for pose detection, pose classification, and rep counting. Integrates with both direct classification and neural network-based classification.
- **NeuralNetwork.py**: Defines the PyTorch neural network model and contains methods for training, testing, and saving model weights.
- **Dataloader.py**: Loads and prepares pose data for neural network training and testing.
- **data_prep.py**: Downloads videos from a list of URLs, extracts frames, and prepares them for model training.

## Future Enhancements
- **Unified Model for Multiple Exercises**: Extend the neural network model to handle a broader range of exercises.
- **Improved Real-Time Feedback**: Implement a more advanced feedback mechanism based on user performance trends.
- **Interactive Interface**: Build a graphical interface to select exercises and access feedback interactively.
