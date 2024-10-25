import torch
import cv2
import numpy as np
from PoseClass import PoseClass
from NeuralNetwork import NeuralNetwork
from DataLoader import Data, DataLoader

def exercize(video_file = 'media/demo-4.mp4'):
    
    detec = PoseClass(image_mode=False, model_complexity=1)
    # video = cv2.VideoCapture('media/'+video_file)
    video = cv2.VideoCapture(video_file)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('demo/output.mp4',fourcc, 20.0, (1300, 700))
    person = 1
    while video.isOpened():
        ok, frame = video.read()

        # Check if frame is not read properly.
        if not ok:
            print('Video finished')
            break
        frame_height, frame_width, _ = frame.shape
        frame = cv2.resize(frame, (1300, 700))
        try:
            frame, landmark = detec.detectPose(frame, display=False)
            frame, label = detec.classifyPose(landmark, frame, display_angles=False)
            frame, reps = detec.countReps(person, label, frame)
        except Exception as e:
            if str(e) != "list index out of range":
                print(e)
        cv2.imshow('Pose Detection', frame)
        out.write(frame)

        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed

        k = cv2.waitKey(1)

        # Check if 'ESC' is pressed.
        if (k == 27):
            # Break the loop.
            break
    detec.saveEx(person)
    print(detec.completed_exercise)
    # Release the VideoCapture object.
    video.release()
    out.release()

    # Close the windows.
    cv2.destroyAllWindows()

def ClassifyPose_NN( nn, landmarks, output_image,labels_strings):
    '''
    This function classifies poses based on a pretrained Neural Network.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    label = ''
    # extract the necessary landmarks and flatten
    if len(landmarks) == 24:
        X = torch.from_numpy(np.array(landmarks).astype(np.float32))

        # get the prediction from the model
        pred = nn.inference(X)

        # get the highest threshold
        index = np.argmax(pred.detach().numpy())

        # reject option, heuristic threshold needs changing
        if pred[index] > 0.6:
            # get the label based on the prediction
            label = labels_strings[index]

        # Write the label on the output image.
    cv2.putText(output_image, 'Pose: ' + label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
    return output_image, label

def exercize_NN(nn,landmarks_tracked,labels_strings,video=0):
    detec = PoseClass(image_mode=False, model_complexity=1)
    video = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('demo/output_nn.mp4',fourcc, 20.0, (1300, 700))
    person = 1
    while video.isOpened():
        ok, frame = video.read()

        # Check if frame is not read properly.
        if not ok:
            print('Video finished')
            break
        frame_height, frame_width, _ = frame.shape
        frame = cv2.resize(frame, (1300, 700))
        try:
            frame, landmark = detec.detectPose_NN(frame,landmarks_tracked)
            frame, label = ClassifyPose_NN(nn,landmark, frame,labels_strings)
            frame, reps = detec.countReps(person, label, frame)
        except Exception as e:
            if str(e) != "list index out of range":
                print(e)
        cv2.imshow('Pose Detection', frame)
        out.write(frame)
        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed

        k = cv2.waitKey(1)

        # Check if 'ESC' is pressed.
        if (k == 27):
            # Break the loop.
            break
    detec.saveEx(person)
    print(detec.completed_exercise)
    # Release the VideoCapture object.
    video.release()
    out.release()
    # Close the windows.
    cv2.destroyAllWindows()

def train_NN(data_file,landmarks_tracked,labels_tracked,landmark_labels,epochs,weights = None, prev_epoch=0):
    print("Loading Data")
    batch_size = 16
    data_file = data_file
    train_data = Data(data_file,landmarks_tracked,labels_tracked,landmark_labels,landmark_file=True,detection_col=True)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    print("Beginning to Train model")
    nn = NeuralNetwork(no_input=24,no_labels=4,train=True, lr = 0.00005,
                       weights=weights,prev_epoch=prev_epoch)

    losses = nn.train_model(train_dataloader,epochs)
    print('Training Complete')
    weights = nn.save_weights(loss = losses[-1],epochs=epochs)
    nn.plot_losses(losses,show=False)
    return nn, weights

def test_NN(data_file , weights,landmarks_tracked,labels_tracked,landmark_labels):
    print("Loading Data")
    batch_size = 1
    data_file = data_file
    test_data = Data(data_file,landmarks_tracked,labels_tracked,landmark_labels,landmark_file=True,detection_col=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    print("Initialise model")
    nn = NeuralNetwork(no_input=24,no_labels=4,train=False, weights=weights)
    y_preds, y_trues, total =nn.test_model(test_dataloader)
    correct = 0
    predictions = []
    y_labels = []
    for i in range(total):
        y_pred = y_preds[i]
        y_true = y_trues[i]
        index_pred = np.argmax(y_pred)
        index_true = np.argmax(y_true)
        predictions.append(index_pred)
        y_labels.append(index_true)
        if index_true == index_pred:
            correct += 1
    print('correct:',correct,'/ Total: ',total)

    nn.get_confusion_matrix(y_labels, predictions)

    return

if __name__ == '__main__':
    # L  R  = position
    # 11 12 = shoulder
    # 13 14 = elbow
    # 15 16 = wrist
    # 23 24 = hip
    # 25 26 = knee
    # 27 28 = ankle
    landmark_labels = ['shoulder_Lx','shoulder_Ly','shoulder_Rx','shoulder_Ry','elbow_Lx','elbow_Ly','elbow_Rx','elbow_Ry','wrist_Lx','wrist_Ly','wrist_Rx','wrist_Ry','hip_Lx','hip_Ly','hip_Rx','hip_Ry','knee_Lx','knee_Ly','knee_Rx','knee_Ry','ankle_Lx','ankle_Ly','ankle_Rx','ankle_Ry']
    landmarks_tracked = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    labels_tracked = { 'standing_hip_abduction_A': 0, 'standing_hip_abduction_B': 1, 'hamstring_curl_A': 2,
                      'hamstring_curl_B': 3,'left_laying_leg_raise_A': 4, 'left_laying_leg_raise_B': 5,}
    labels_strings = ['Standing', 'Hip Abduction', 'Standing',
                      'Hamstring Curl' ]

    data_file = './exercise_2_vids_40.csv'
    dataset = './dataset.csv'
    weights = './weights/'+'2023-04-27_20-45-10_classes_4_epochs_3000_loss_0.7450538873672485.pth'
    # weights = None
    # epochs = 3000
    # prev_epoch = 0
    # nn, weights = train_NN(dataset, landmarks_tracked, labels_tracked,landmark_labels, epochs,weights=weights,prev_epoch=prev_epoch)
    # test_NN(dataset, weights, landmarks_tracked, labels_tracked,landmark_labels)

    video_file = 'media/demo-4.mp4'
    # video_file = 0 # webcam
    #
    nn = NeuralNetwork(no_input=24,no_labels=4,train=False,weights=weights)
    print('running exercise')
    exercize_NN(nn,landmarks_tracked,labels_strings,video=video_file)

    # exercize()

    # TUGtest()
