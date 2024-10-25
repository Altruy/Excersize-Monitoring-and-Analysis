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

# Function to check if the leg is lifted (marching)
def check_leg_lift(landmarks):
    shoulders = get_landmark_coords(landmarks, 'LEFT_SHOULDER')
    hip_left = get_landmark_coords(landmarks, 'LEFT_HIP')
    hip_right = get_landmark_coords(landmarks, 'RIGHT_HIP')
    knee_left = get_landmark_coords(landmarks, 'LEFT_KNEE')
    knee_right = get_landmark_coords(landmarks, 'RIGHT_KNEE')
    ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')
    
    hip_angle_left = calculate_angle(shoulders, hip_left, knee_left)  # Use vertical axis for the first point
    hip_angle_right = calculate_angle(shoulders, hip_right, knee_right)

    if 100 > hip_angle_right > 90 or 100 > hip_angle_left > 90:  # Adjust threshold for detecting lift
        return "Lifted"
    else:
        return "Down"

# Function to evaluate leg lift position and provide feedback
def evaluate_leg_lift_position(landmarks):        
    shoulders = get_landmark_coords(landmarks, 'LEFT_SHOULDER')
    hip_left = get_landmark_coords(landmarks, 'LEFT_HIP')
    hip_right = get_landmark_coords(landmarks, 'RIGHT_HIP')
    knee_left = get_landmark_coords(landmarks, 'LEFT_KNEE')
    knee_right = get_landmark_coords(landmarks, 'RIGHT_KNEE')

    hip_angle_left = calculate_angle(shoulders, hip_left, knee_left)  # Use vertical axis for the first point
    hip_angle_right = calculate_angle(shoulders, hip_right, knee_right)
    feedback = "Good lift"
    if hip_angle_left >= 90 or hip_angle_right >= 90:
        feedback = "Lift your leg higher"
    elif hip_angle_right > 100 or hip_angle_left > 100:
        feedback = "Lower your leg"
    
    return feedback, hip_angle_left, hip_angle_right

# Specify the video file path here
# video_file_path = 'files/seated_marching.mp4'

# Main function to process the video file
def process_video(camera_index): 
    cap = cv2.VideoCapture(camera_index)
    
    march_count = 0
    last_position = None

    while cap.isOpened(): 
        ret, frame = cap.read() 
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            position = check_leg_lift(landmarks)
            feedback = ""  # Initialize feedback variable
            cv2.resize(frame, (800, 600))

            if position == "Lifted":
                feedback, hip_angle_left, hip_angle_right = evaluate_leg_lift_position(landmarks)
            else:
                feedback = "Good Pose"
                hip_angle_right = hip_angle_left = 90  # Default angle when the leg is down


            if last_position == "Down" and position == "Lifted":
                march_count += 1/2

            last_position = position

            cv2.putText(frame, f'Marches: {march_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Hip Angle Left: {hip_angle_left:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Hip Angle Right: {hip_angle_right:.2f}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, feedback, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Position: {position}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow('Seated Marching Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process the specified video file
process_video(0)
