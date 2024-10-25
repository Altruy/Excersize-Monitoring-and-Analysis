import math
import cv2
from time import time
import mediapipe as mp

class PoseClass:
    def __init__(self,input_landmarks = 36,total_classes=12, image_mode=True, min_detection_confidence=0.3,model_complexity=2):
        self.mp_pose = mp.solutions.pose
        # Setting up the Pose function.
        self.pose = self.mp_pose.Pose(static_image_mode=image_mode, min_detection_confidence=min_detection_confidence,model_complexity=model_complexity)

        # Initializing mediapipe drawing class, useful for annotation.
        self.mp_drawing = mp.solutions.drawing_utils

        self.ac_log = {}
        self.completed_exercise = {}
        self.tug_state = 0
        self.tug_label = 'test not ready'
        self.last_tug_time = 0
        self.tug_time = 0
        self.color = (0, 0, 0)

        self.input_landmarks = input_landmarks
        self.total_classes = total_classes
        # self.ex_model = NeuralNetwork(no_input = self.input_landmarks, no_labels = self.total_classes)


    def detectPose(self, image, display=False):
        '''
        This function performs pose detection on an image.
        Args:
            image: The input image with a prominent person whose pose landmarks needs to be detected.
            pose: The pose setup function required to perform the pose detection.
            display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                     and the pose landmarks in 3D plot and returns nothing.
        Returns:
            output_image: The input image with the detected pose landmarks drawn.
            landmarks: A list of detected landmarks converted into their original scale.
        '''
        # Create a copy of the input image.
        output_image = image.copy()

        # Convert the image from BGR into RGB format.
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform the Pose Detection.
        results = self.pose.process(imageRGB)

        # Retrieve the height and width of the input image.
        height, width, _ = image.shape

        # Initialize a list to store the detected landmarks.
        landmarks = []

        # Check if any landmarks are detected.
        if results.pose_landmarks:

            # Draw Pose landmarks on the output image.
            self.mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                      connections=self.mp_pose.POSE_CONNECTIONS)

            # Iterate over the detected landmarks.
            for landmark in results.pose_landmarks.landmark:
                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

        # Check if the original input image and the resultant image are specified to be displayed.
        if display:

            # Display the original input image and the resultant image.
            # plt.figure(figsize=[22, 22])
            # plt.subplot(121);
            # plt.imshow(image[:, :, ::-1]);
            # plt.title("Original Image");
            # plt.axis('off');
            # plt.subplot(122);
            # plt.imshow(output_image[:, :, ::-1]);
            # plt.title("Output Image");
            # plt.axis('off');

            # Also Plot the Pose landmarks in 3D.
            self.mp_drawing.plot_landmarks(results.pose_world_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Otherwise
        else:

            # Return the output image and the found landmarks.
            return output_image, landmarks

    def calculateAngle(self,landmark1, landmark2, landmark3):
        '''
        This function calculates angle between three different landmarks.
        Args:
            landmark1: The first landmark containing the x,y and z coordinates.
            landmark2: The second landmark containing the x,y and z coordinates.
            landmark3: The third landmark containing the x,y and z coordinates.
        Returns:
            angle: The calculated angle between the three landmarks.

        '''

        # Get the required landmarks coordinates.
        x1, y1, _ = landmark1
        x2, y2, _ = landmark2
        x3, y3, _ = landmark3

        # Calculate the angle between the three points
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        # Check if the angle is less than zero.
        if angle < 0:

            # Add 360 to the found angle.
            angle += 360

        # Return the calculated angle.
        return angle

    def classifyPose(self, landmarks, output_image, display_angles=False):
        '''
        This function classifies poses depending upon the angles of various body joints.
        Args:
            landmarks: A list of detected landmarks of the person whose pose needs to be classified.
            output_image: A image of the person with the detected pose landmarks drawn.
            display: A boolean value that is if set to true the function displays the resultant image with the pose label
            written on it and returns nothing.
        Returns:
            output_image: The image with the detected pose landmarks drawn and pose label written.
            label: The classified pose label of the person in the output_image.

        '''

        # Initialize the label of the pose. It is not known at this stage.
        label = ''

        # Calculate the required angles.
        # ----------------------------------------------------------------------------------------------------------------

        # Get the angle between the left shoulder, elbow and wrist points.
        left_elbow_angle = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value])

        # Get the angle between the right shoulder, elbow and wrist points.
        right_elbow_angle = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                           landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value])

        # Get the angle between the left elbow, shoulder and hip points.
        left_shoulder_angle = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value])

        # Get the angle between the right hip, shoulder and elbow points.
        right_shoulder_angle = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                              landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                              landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value])

        # Get the angle between the left hip, knee and ankle points.
        left_knee_angle = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                                         landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])

        # Get the angle between the right hip, knee and ankle points
        right_knee_angle = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])

        right_shld_hip_knee = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value])

        left_shld_hip_knee = 360 - self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                  landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                                  landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value])

        waist_x = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][0] - landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][0]
        waist_x = math.sqrt(waist_x*waist_x)
        waist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][1] - landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][1]
        waist_y = math.sqrt(waist_y*waist_y)

        r_l_str =  right_knee_angle > 165 and right_knee_angle < 215
        l_l_str = left_knee_angle > 165 and left_knee_angle < 215
        # ----------------------------------------------------------------------------------------------------------------
        # Check if standing Straight
        # Check if it is the warrior II pose or the T pose.
        # As for both of them, both arms should be straight and shoulders should be at the specific angle.
        # ----------------------------------------------------------------------------------------------------------------
        # Leg conditions:
        if (right_shld_hip_knee > 250 or right_shld_hip_knee < 110) and (
                left_shld_hip_knee > 250 or left_shld_hip_knee < 110) and (
                right_knee_angle > 220 or right_knee_angle < 120) and (
                left_knee_angle > 220 or left_knee_angle < 120):
            label = "Sitting"
        if waist_x > waist_y:
            if right_shld_hip_knee < 200 and left_shld_hip_knee < 200 and r_l_str and l_l_str:
                # add arm condition??
                label = "Standing"



            if (right_shld_hip_knee > 210 or right_shld_hip_knee < 150) and left_shld_hip_knee < 200  and r_l_str and l_l_str:
                label = "Right Hip Abduction"

            elif right_shld_hip_knee < 200 and (left_shld_hip_knee > 210 or left_shld_hip_knee < 150) and r_l_str and l_l_str:
                label = "Left Hip Abduction"

            if right_shld_hip_knee < 200 and left_shld_hip_knee < 200 and l_l_str and (
                    right_knee_angle > 300 or right_knee_angle < 150):
                label = "Right Hamstring Curl"

            elif right_shld_hip_knee < 200 and left_shld_hip_knee < 200 and r_l_str and (
                    left_knee_angle > 300 or left_knee_angle < 150):
                label = "Left Hamstring Curl"

        else:
            if right_shld_hip_knee < 200 and left_shld_hip_knee < 200 and r_l_str and l_l_str:
                # add arm condition??
                label = "Laying"

            if (right_shld_hip_knee > 210 or right_shld_hip_knee < 150) and left_shld_hip_knee < 200 and r_l_str and l_l_str:
                label = "Right Leg Raise"

            elif right_shld_hip_knee < 200 and (left_shld_hip_knee > 210 or left_shld_hip_knee < 150) and r_l_str and l_l_str:
                label = "Left Leg Raise"

            if (right_shld_hip_knee > 210 or right_shld_hip_knee < 150) and left_shld_hip_knee < 200 and (
                    right_knee_angle > 275 or right_knee_angle < 110) :
                label = "Right Heel Slide"

            elif right_shld_hip_knee < 200 and (left_shld_hip_knee > 210 or left_shld_hip_knee < 150) and (
                    left_knee_angle > 275 or left_knee_angle < 110):
                label = "Left Heel Slide"




        # ----------------------------------------------------------------------------------------------------------------


            # Write the label on the output image.
        cv2.putText(output_image, 'Pose: '+label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, self.color, 2)

        # Check if the resultant image is specified to be displayed.
        if display_angles:
            cv2.putText(output_image,'L Sh Hip Hl:'+str(left_shld_hip_knee)[:5],(10, 60), cv2.FONT_HERSHEY_PLAIN, 2, self.color, 2)
            cv2.putText(output_image,'R Sh Hip Hl:'+str(right_shld_hip_knee)[:5],(550, 60), cv2.FONT_HERSHEY_PLAIN, 2, self.color, 2)
            cv2.putText(output_image,'left knee:'+str(left_knee_angle)[:5],(10, 90), cv2.FONT_HERSHEY_PLAIN, 2, self.color, 2)
            cv2.putText(output_image,'right knee:'+str(right_knee_angle)[:5],(550, 90), cv2.FONT_HERSHEY_PLAIN, 2, self.color, 2)
        return output_image, label

    def countReps(self, person, label, frame, ):
        if person not in self.ac_log:
            self.ac_log[person] = { 'log': [label],'done_ex' : [], 'current_ex': 'None', 'count': 0}
        else:
            try:
                current_ex = self.ac_log[person]['current_ex']
                last_step = self.ac_log[person]['log'][-1]

                # new exercise starting
                # curent == None
                # current ==  same as label
                # current == diferent
                if label != last_step and label != '':
                    if label != 'Standing' and label != 'Laying' and label != '':
                        # starting new exercise saving old in progress log
                        if current_ex != label and current_ex != 'None':
                            print('Starting new','old',current_ex,'new',label)
                            self.saveEx(person)

                        self.ac_log[person]['current_ex'] = label

                    if (label == 'Standing' or label == 'Laying') and current_ex == last_step:
                        self.ac_log[person]['count'] += 1
                    self.ac_log[person]['log'].append(label)

                if self.ac_log[person]['current_ex'] == 'None':
                    color = (0, 0, 0)
                else:
                    color = (0, 0, 0)

                cv2.putText(frame,'Exercise: '+ self.ac_log[person]['current_ex'],(10, 630), cv2.FONT_HERSHEY_PLAIN, 2, self.color, 2)
                cv2.putText(frame,'Reps: '+ str(self.ac_log[person]['count']),(10, 660), cv2.FONT_HERSHEY_PLAIN, 2, self.color, 2)
            except Exception as e:
                print('Count excepton',e)
        return frame, self.ac_log[person]['count']

    def saveEx(self,person):
        if person in self.ac_log:
            if self.ac_log[person]['count'] > 0:
                if person in self.completed_exercise:
                    self.completed_exercise[person].append({'exercise': self.ac_log[person]['current_ex'], 'count': self.ac_log[person]['count']})
                else:
                    self.completed_exercise[person] = [{'exercise': self.ac_log[person]['current_ex'], 'count': self.ac_log[person]['count']}]
                self.ac_log[person]['count'] = 0

    def TUGtest(self, landmarks, output_image):
        '''
        This function classifies yoga poses depending upon the angles of various body joints.
        Args:
            landmarks: A list of detected landmarks of the person whose pose needs to be classified.
            output_image: A image of the person with the detected pose landmarks drawn.
            display: A boolean value that is if set to true the function displays the resultant image with the pose label
            written on it and returns nothing.
        Returns:
            output_image: The image with the detected pose landmarks drawn and pose label written.
            label: The classified pose label of the person in the output_image.

        '''

        # Initialize the label of the pose. It is not known at this stage.
        label = ''

        # Calculate the required angles.
        # ----------------------------------------------------------------------------------------------------------------

        # Get the angle between the left hip, knee and ankle points.
        left_knee_angle = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                              landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                                              landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])

        # Get the angle between the right hip, knee and ankle points
        right_knee_angle = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                               landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                               landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])

        right_shld_hip_knee = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value])

        left_shld_hip_knee = 360 - self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                       landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                                       landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value])

        waist_x = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][0] - \
                  landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][0]
        waist_x = math.sqrt(waist_x * waist_x)
        waist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][1] - \
                  landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][1]
        waist_y = math.sqrt(waist_y * waist_y)

        r_l_str = right_knee_angle > 165 and right_knee_angle < 215
        l_l_str = left_knee_angle > 165 and left_knee_angle < 215
        # ----------------------------------------------------------------------------------------------------------------
        # Check if standing Straight
        # Check if it is the warrior II pose or the T pose.
        # As for both of them, both arms should be straight and shoulders should be at the specific angle.
        # ----------------------------------------------------------------------------------------------------------------
        # Leg conditions:
        if (right_shld_hip_knee > 250 or right_shld_hip_knee < 110) and (
                left_shld_hip_knee > 250 or left_shld_hip_knee < 110) and (
                right_knee_angle > 220 or right_knee_angle < 120) and (
                left_knee_angle > 220 or left_knee_angle < 120):
            label = "Sitting"
        if waist_x > waist_y:
            if right_shld_hip_knee < 200 and left_shld_hip_knee < 200 and r_l_str and l_l_str:
                # add arm condition??
                label = "Standing"


        # ----------------------------------------------------------------------------------------------------------------

        if label == "Sitting":
            if self.tug_state == 0:
                self.tug_state = 1
                self.tug_label = "Start walking to start"

            elif self.tug_state == 2:
                self.tug_state = 3
                self.last_tug_time = time() - self.tug_time
                self.tug_label = "Test completed"

        elif label == "Standing":
            if self.tug_state == 1:
                # walking started
                self.tug_state = 2
                self.tug_time = time()
                self.tug_label = "Test started"

            # Write the label on the output image.
        cv2.putText(output_image, 'Pose: ' + label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, self.color, 2)

        # Check if the resultant image is specified to be displayed.

        cv2.putText(output_image, self.tug_label, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2,
                    self.color, 2)
        if self.tug_state == 3:
            cv2.putText(output_image, "TUG time:"+str(self.last_tug_time)[:4], (10, 90), cv2.FONT_HERSHEY_PLAIN, 2,
                    self.color, 2)

        return output_image, label

    def detectPose_NN(self, image,landmark_indexes, display=False):
        '''
        This function performs pose detection on an image.
        Args:
            image: The input image with a prominent person whose pose landmarks needs to be detected.
            pose: The pose setup function required to perform the pose detection.
            display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                     and the pose landmarks in 3D plot and returns nothing.
        Returns:
            output_image: The input image with the detected pose landmarks drawn.
            landmarks: A list of detected landmarks converted into their original scale.
        '''
        # Create a copy of the input image.
        output_image = image.copy()

        # Convert the image from BGR into RGB format.
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform the Pose Detection.
        results = self.pose.process(imageRGB)

        # Retrieve the height and width of the input image.
        height, width, _ = image.shape

        # Initialize a list to store the detected landmarks.
        landmarks = []

        # Check if any landmarks are detected.
        if results.pose_landmarks:

            # Draw Pose landmarks on the output image.
            self.mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                      connections=self.mp_pose.POSE_CONNECTIONS)

            # Iterate over the detected landmarks.
            for index in landmark_indexes:
                try:
                    landmarks.append(int(results.pose_landmarks.landmark[index].x * width))
                    landmarks.append(int(results.pose_landmarks.landmark[index].y * height))
                except:
                    landmarks.append(0)
                    landmarks.append(0)

        return output_image, landmarks


def oneframe(fil):
    detec = PoseClass(image_mode=False, model_complexity=1)
    # video = cv2.VideoCapture('media/'+video_file)
    frame = cv2.imread(fil)
    frame_height, frame_width, _ = frame.shape
    frame = cv2.resize(frame, (1200, 700))
    try:
        frame, landmark = detec.detectPose(frame, display=False)
        # frame, label = detec.classifyPose(landmark, frame, display_angles=True)
        # frame, reps = detec.countReps(person, label, frame)
    except Exception as e:
        if str(e) != "list index out of range":
            print(e)
    cv2.imshow('Pose Detection', frame)
    k = cv2.waitKey(100000000)

if __name__ == '__main__':
    oneframe('frame1.jpg')
    oneframe('frame2.jpg')