import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Function to get the coordinates of a specific landmark
def get_landmark_coords(landmarks, landmark_name):
    landmark = landmarks[mp_pose.PoseLandmark[landmark_name].value]
    return [landmark.x, landmark.y]

# Function to check the current position (Straight or Bent)
def check_position(landmarks):
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    elbow = get_landmark_coords(landmarks, 'LEFT_ELBOW')
    wrist = get_landmark_coords(landmarks, 'LEFT_WRIST')

    hip_angle = calculate_angle(shoulder, hip, knee)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    if hip_angle >= 80 or elbow_angle < 170:
        return "Straight"
    else:
        return "Bent"

# Function to evaluate straight position and provide feedback
def evaluate_straight_position(landmarks):
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    elbow = get_landmark_coords(landmarks, 'LEFT_ELBOW')
    wrist = get_landmark_coords(landmarks, 'LEFT_WRIST')

    hip_angle = calculate_angle(shoulder, hip, knee)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    feedback = "Good posture"
    if hip_angle > 100:
        feedback = "Sit up straight"

    return feedback, hip_angle, elbow_angle

# Function to evaluate bent position and provide feedback
def evaluate_bent_position(landmarks):
    shoulder = get_landmark_coords(landmarks, 'LEFT_SHOULDER')
    hip = get_landmark_coords(landmarks, 'LEFT_HIP')
    knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
    elbow = get_landmark_coords(landmarks, 'LEFT_ELBOW')
    wrist = get_landmark_coords(landmarks, 'LEFT_WRIST')

    hip_angle = calculate_angle(shoulder, hip, knee)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    feedback = "Good posture"
    if hip_angle > 120:
        feedback = "Bend more."
    elif hip_angle < 60:
        feedback = "Bend less."

    if elbow_angle < 150:
        feedback += " Straighten your elbows"

    print(f"Bent Position: Hip angle = {hip_angle}, Elbow angle = {elbow_angle}, Feedback = {feedback}")  # Debug statement

    return feedback, hip_angle, elbow_angle

# Specify the video file path here
# video_file_path = 'files/hamstring.mp4'

# Main function to process the video file
def process_video(camera_index):
    cap = cv2.VideoCapture(camera_index)

    counter = 0
    last_position = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            position = check_position(landmarks)
            feedback = ""  # Initialize feedback variable

            # Resize the frame for better visibility
            frame = cv2.resize(frame, (800, 600))

            if position == "Straight":
                feedback, hip_angle, elbow_angle = evaluate_straight_position(landmarks)
            elif position == "Bent":
                feedback, hip_angle, elbow_angle = evaluate_bent_position(landmarks)
            else:
                feedback = "Transitioning"
                hip_angle = elbow_angle = 0

            if last_position == "Bent" and position == "Straight":
                counter += 1

            last_position = position 

            # Display angles and feedback on the frame
            cv2.putText(frame, f'Counts: {counter}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Position: {position}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Hip Angle: {hip_angle:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Elbow Angle: {elbow_angle:.2f}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Seated Hamstring Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process the specified video file
process_video(0)
