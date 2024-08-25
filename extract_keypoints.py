import pandas as pd
import mediapipe as mp
import cv2
import os
import multiprocessing
from joblib import Parallel, delayed
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.pose import PoseLandmark
from collections import defaultdict
from typing import Dict, List
import random
import contextlib
import sys
import numpy as np
from multiprocessing import Lock, Manager, Pool
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
holistic = mp_holistic.Holistic()
hand_landmarks = ['INDEX_FINGER_DIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_TIP', 
                  'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_TIP', 
                  'PINKY_DIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_TIP', 'RING_FINGER_DIP', 'RING_FINGER_MCP', 
                  'RING_FINGER_PIP', 'RING_FINGER_TIP', 'THUMB_CMC', 'THUMB_IP', 'THUMB_MCP', 'THUMB_TIP', 'WRIST']
pose_landmarks = ['LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 
                  'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_PINKY', 
                  'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_WRIST', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE', 
                  'RIGHT_ANKLE', 'RIGHT_EAR', 'RIGHT_ELBOW', 'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER', 
                  'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY', 
                  'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST']

manager = Manager()
lock = manager.Lock()

class DummyFile(object):
  file = None
  def __init__(self, file):
    self.file = file

  def write(self, x):
    # Avoid print() second call (useless \n)
    if len(x.rstrip()) > 0:
        tqdm.write(x, file=self.file)

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout

def file_name(mode):
    file_path = "wsl100.csv"  # Replace with the actual file path
    print(file_path)
    train_path = []
    labels = []
    try:
        data = pd.read_csv(file_path)
        for line in data["file"]:
            # Process each line here
            # print(line.strip())  # Example: Print the line after removing leading/trailing whitespaces
            path = "00 SLR/archive/videos/"
            train_path.append(path + line.strip())
            # search_string(line.strip(), "extra")
        for label in data["label"]:
            # print(label)
            labels.append(int(label))
    except FileNotFoundError:
        print("File not found.")
    except IOError:
        print("Error reading the file.")
    return train_path, labels

def extract_keypoint(video_path, label, debug):
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    count = 0
    pose_start = 0
    pose_end = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        keypoint_dict : Dict[str, List[float]] = defaultdict(list)
        while True:
            ret, frame = cap.read()
            count += 1

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = image.shape[0], image.shape[1]

            if count == 1:
                results = holistic.process(image)
                if not results.pose_landmarks:
                    pose_start = 0
                    pose_end = width
                else:
                    # Crop pose only 
                    print("Right shoulder", results.pose_landmarks.landmark[12].x)
                    print("Left shoulder", results.pose_landmarks.landmark[11].x)
                    width_man = abs(results.pose_landmarks.landmark[12].x - results.pose_landmarks.landmark[11].x)
                    scale = 0.5
                    pose_start = int(width * (results.pose_landmarks.landmark[12].x - scale * width_man)) 
                    pose_end = int(width * (results.pose_landmarks.landmark[11].x + scale * width_man)) 
                    print(width_man)
                    print(pose_start)
                    print(pose_end)
            image = image[0:height, pose_start:pose_end]
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Smoothing image
            image = cv2.GaussianBlur(image, (5, 5), 0)            
            if debug:
                directory = f"pre_processing/{video_path}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    with nostdout():
                        print(f"Directory '{directory}' created.")
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                cv2.imwrite(f"{directory}/keypoint_{count}.jpg", image)

                black = np.zeros(image.shape, dtype=np.uint8)
                mp_drawing.draw_landmarks(black, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(black, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(black, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                cv2.imwrite(f"{directory}/keypoint_black_{count}.jpg", black)

            if results.right_hand_landmarks:
                for idx, landmark in enumerate(results.right_hand_landmarks.landmark): 
                    keypoint_dict[f"{hand_landmarks[idx]}_right_x"].append(landmark.x)
                    keypoint_dict[f"{hand_landmarks[idx]}_right_y"].append(landmark.y)
                    keypoint_dict[f"{hand_landmarks[idx]}_right_z"].append(landmark.z)
            else:
                for idx in range(len(hand_landmarks)):
                    keypoint_dict[f"{hand_landmarks[idx]}_right_x"].append(0)
                    keypoint_dict[f"{hand_landmarks[idx]}_right_y"].append(0)
                    keypoint_dict[f"{hand_landmarks[idx]}_right_z"].append(0)

            if results.left_hand_landmarks:
                for idx, landmark in enumerate(results.left_hand_landmarks.landmark): 
                    keypoint_dict[f"{hand_landmarks[idx]}_left_x"].append(landmark.x)
                    keypoint_dict[f"{hand_landmarks[idx]}_left_y"].append(landmark.y)
                    keypoint_dict[f"{hand_landmarks[idx]}_left_z"].append(landmark.z)
            else:
                for idx in range(len(hand_landmarks)):
                    keypoint_dict[f"{hand_landmarks[idx]}_left_x"].append(0)
                    keypoint_dict[f"{hand_landmarks[idx]}_left_y"].append(0)
                    keypoint_dict[f"{hand_landmarks[idx]}_left_z"].append(0)

            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark): 
                    keypoint_dict[f"{pose_landmarks[idx]}_x"].append(landmark.x)
                    keypoint_dict[f"{pose_landmarks[idx]}_y"].append(landmark.y)
                    keypoint_dict[f"{pose_landmarks[idx]}_z"].append(landmark.z)
            else:
                for idx in range(len(pose_landmarks)):
                    keypoint_dict[f"{pose_landmarks[idx]}_x"].append(0)
                    keypoint_dict[f"{pose_landmarks[idx]}_y"].append(0)
                    keypoint_dict[f"{pose_landmarks[idx]}_z"].append(0)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        keypoint_dict["frame"] = count
        keypoint_dict["video_path"] = video_path
        keypoint_dict["label"] = label

        values_list = [[value] for value in keypoint_dict.values()]
        # Create the DataFrame
        df = pd.DataFrame(values_list, index=keypoint_dict.keys())
        df = df.transpose()
        csv_file_path = f"output_keypoints_wsl_100_holistic(v2).csv" # Replace with the name you want 
        # Write the DataFrame to a CSV file 
        df.to_csv(csv_file_path, mode='a', header=False, index=False)

if __name__ == '__main__':
    from tqdm import tqdm
    modes = ["train", "valid", "test"]
    n_cores = 4
    
    for mode in modes:
        train_path, labels = file_name(mode)
        print(f"Len {mode} : {len(train_path)}")
        with tqdm(total=len(train_path), file=sys.stdout) as t:
            ### Sequential code
            # for video_path, label in zip(train_path, labels):
            #     extract_keypoint(video_path, label, random.random() >= 0.8)
            #     t.update(1)
            ### Parrallel code
            Parallel(n_jobs=n_cores)(
                delayed(extract_keypoint)(video_path, label, random.random() >= 0.8)
                for video_path, label in zip(train_path, labels)
            )
            t.update(1)