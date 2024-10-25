import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PoseClass import PoseClass
import cv2

# landmarks = https://www.mdpi.com/applsci/applsci-13-02700/article_deploy/html/images/applsci-13-02700-g001-550.jpg

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, file, landmarks_tracked, labels_tracked,landmark_labels,landmark_file = False,write_frames=False,detection_col=False):
        self.poseClass = PoseClass()
        self.landmark_labels =landmark_labels
        self.landmarks_tracked = landmarks_tracked
        self.labels_tracked = labels_tracked
        self.detect_col = detection_col
        self.write_frames = write_frames
        self.df = pd.read_csv(file)
        self.classes = pd.unique(self.df['label'])

        if landmark_file:
            self.X, self.y = self.poses_from_csv()
        else:
            self.X, self.y = self.convert_files_poses()
        self.len = self.X.shape[0]

    def poses_from_csv(self):
        X = self.df.iloc[:,1:25]
        y = self.df.iloc[:,25]
        Ys = []
        for i in y:
            label_onehot = [0 for _ in range(len(self.classes))]
            label_onehot[i] = 1
            Ys.append(np.array(label_onehot))
        return torch.from_numpy(np.array(X).astype(np.float32)), torch.from_numpy(np.array(Ys).astype(np.float32))


    def convert_files_poses (self):
        X = []
        y = []
        x_y = []
        try:
            os.mkdir('./train_frames')
        except:
            pass
        for i in range(len(self.classes)):
            try:
                os.mkdir('./train_frames/'+str(i))
            except:
                pass
        for index, row in self.df.iterrows():
            path = './'+str(row['path']+'.jpg')
            label = row['label']
            if self.detect_col:
                if row['detection'] == 'no':
                    # print('no pose for',path)
                    continue
            im = cv2.imread(path)
            if im is not None:
                im_, landmarks = self.poseClass.detectPose_NN(im,self.landmarks_tracked)
                if len(landmarks) == 24:
                    label_onehot = [0 for _ in range(len(self.classes))]
                    label_onehot[self.labels_tracked[label]] = 1
                    y.append(np.array(label_onehot))
                    X.append(np.array(landmarks))
                    x_y.append(np.append(landmarks,[self.labels_tracked[label]]))
                    # write frames to see
                    if self.write_frames:
                        frame_name = '_'.join(path[1:].split('/'))
                        cv2.imwrite('./train_frames/'+str(self.labels_tracked[label])+'/'+str(index)+frame_name,im_)

        cols = self.landmark_labels + ['label']
        df = pd.DataFrame(x_y,columns=cols)
        df.to_csv('./dataset.csv')

        return torch.from_numpy(np.array(X).astype(np.float32)), torch.from_numpy(np.array(y).astype(np.float32))






    def __getitem__(self, index):
        # open the frame and process it
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    # L  R  = position
    # 11 12 = shoulder
    # 13 14 = elbow
    # 15 16 = wrist
    # 23 24 = hip
    # 25 26 = knee
    # 27 28 = ankle
    landmark_labels = ['shoulder_Lx', 'shoulder_Ly', 'shoulder_Rx', 'shoulder_Ry', 'elbow_Lx', 'elbow_Ly', 'elbow_Rx',
                       'elbow_Ry', 'wrist_Lx', 'wrist_Ly', 'wrist_Rx', 'wrist_Ry', 'hip_Lx', 'hip_Ly', 'hip_Rx',
                       'hip_Ry', 'knee_Lx', 'knee_Ly', 'knee_Rx', 'knee_Ry', 'ankle_Lx', 'ankle_Ly', 'ankle_Rx',
                       'ankle_Ry']
    landmarks_tracked = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    labels_tracked = {'standing_hip_abduction_A': 0, 'standing_hip_abduction_B': 1, 'hamstring_curl_A': 2,
                      'hamstring_curl_B': 3, 'left_laying_leg_raise_A': 4, 'left_laying_leg_raise_B': 5, }

    data_file = './exercise_2_vids_40.csv'
    batch_size = 8
    # Instantiate training and test data
    train_data = Data(data_file,landmarks_tracked,labels_tracked,landmark_labels,write_frames=True,detection_col=True)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # test_data = Data(X_test, y_test)
    # test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Check it's working
    for batch, (X, y) in enumerate(train_dataloader):
        print(f"Batch: {batch + 1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        break

    """
    Batch: 1
    X shape: torch.Size([8, 2])
    y shape: torch.Size([8])
    """