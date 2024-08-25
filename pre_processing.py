import pandas as pd
import ast
import numpy as np
from tqdm import tqdm

def find_index(array):
    for i, num in enumerate(array):
        if num != 0:
            return i

def curl_skeleton(array):
    # print("Original:")
    # print(array)
    if sum(array) == 0:
        return array
    for i, location in enumerate(array):
        # Khác 0 là ko xử lý gì
        if location != 0:
            continue
        # Bằng 0 mới xử lý
        else:
            # Nếu là số đầu tiên và số cuối ko xử lý
            if i == 0 or i == len(array) - 1:
                continue
            # Xử lý số ở giữa
            else:
                # Nếu số ở bên phải khác 0 -> thực hiện nội suy
                if array[i + 1] != 0:
                    # print("From ", array[i], end = " ")
                    array[i] = float((array[i-1]+array[i+1])/2)
                    # print("-> ", array[i])
                # Nếu số ở bên phải bằng 0
                else:
                    # Trong trường hợp tất cả các số còn lại là 0 -> ko xử lý
                    if sum(array[i:]) == 0:
                        continue
                    # Nếu tồn tại số khác 0 -> tìm vị trí số ấy và thực hiện nội suy
                    else:
                        # print("From ", array[i], end = " ")
                        j = find_index(array[i+1:])
                        array[i] = float(((1+j)*array[i-1] + 1*array[i + 1 + j])/(2+j))
                        # print("-> ", array[i])
    # print("After:")
    # print(array)
    return array

if __name__ == "__main__":
    df = pd.read_csv("output_keypoints_wsl_100_holistic(v2).csv") # Replace path as you want
    df = df.set_index('video_path', inplace=False)

    hand_landmarks = ['INDEX_FINGER_DIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_TIP', 
                  'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_TIP', 
                  'PINKY_DIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_TIP', 'RING_FINGER_DIP', 'RING_FINGER_MCP', 
                  'RING_FINGER_PIP', 'RING_FINGER_TIP', 'THUMB_CMC', 'THUMB_IP', 'THUMB_MCP', 'THUMB_TIP', 'WRIST']
    
    HAND_IDENTIFIERS = [id + "_right" for id in hand_landmarks] + [id + "_left" for id in hand_landmarks]

    POSE_IDENTIFIERS = ["RIGHT_SHOULDER", "LEFT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW"]

    body_identifiers = HAND_IDENTIFIERS + POSE_IDENTIFIERS 
    print(body_identifiers)
    print(len(body_identifiers))
    frames = 80
    modes = ["train", "valid", "test"]
    for mode in modes:
        print(f"wsl_100_{mode}_center(3).csv") # Replace name as you want
        train_data = pd.read_csv(f"wsl_100_{mode}_center(3).csv")
        print(len(train_data))
        # B, N, T, V, 1
        data = []
        labels = []
        path = "00 SLR/archive/videos/" # replace path as you want
        for video_index, video in tqdm(train_data.iterrows()):
            row_index = path + video["file"]
            print(row_index)
            row = df.loc[row_index]
            T = len(ast.literal_eval(row["INDEX_FINGER_DIP_right_x"]))
            current_row = np.empty(shape=(2, T, len(body_identifiers), 1))
            for index, identifier in enumerate(body_identifiers):
                    data_keypoint_preprocess_x = curl_skeleton(ast.literal_eval(row[identifier + "_x"]))
                    current_row[0, :, index, :] = np.asarray(data_keypoint_preprocess_x).reshape(T, 1)
                    data_keypoint_preprocess_y = curl_skeleton(ast.literal_eval(row[identifier + "_y"]))
                    current_row[1, :, index, :] = np.asarray(data_keypoint_preprocess_y).reshape(T, 1)

            if T < frames:
                target = np.zeros(shape=(2, frames, len(body_identifiers), 1))
                target[:, :T, :, :] = current_row
            else:
                target = np.zeros(shape=(2, frames, len(body_identifiers), 1))
                target = current_row[:, :frames, :, :]
            data.append(target)
            labels.append(int(row["label"]))

        keypoint_data = np.stack(data, axis=0)
        label_data = np.stack(labels, axis=0)
        np.save(f'wsl100_{mode}_data_preprocess(4).npy', keypoint_data)
        np.save(f'wsl100_{mode}_label_preprocess(4).npy', label_data)
        print("ok")