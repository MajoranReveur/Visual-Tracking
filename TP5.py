from keras.src.applications import ResNet50
from scipy.optimize import linear_sum_assignment
from tensorflow.keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
import cv2
import numpy as np
import pandas as pd
from KalmanFilter import KalmanFilter
import time

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet')
embedding_layer = base_model.get_layer('avg_pool')  # 'avg_pool' is the Global Average Pooling layer
embedding_model = Model(inputs=base_model.input, outputs=embedding_layer.output)

# Function to obtain ResNet embeddings for a region in an image
def get_resnet_embedding(image, box):
    x, y, w, h = int(box.bb_left), int(box.bb_top), int(box.bb_width), int(box.bb_height)
    patch = image[y:y+h, x:x+w]
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return None
    patch = cv2.resize(patch, (224, 224))  # adjust the size based on your model
    patch = preprocess_input(patch)
    patch = np.expand_dims(patch, axis=0)
    embedding = embedding_model.predict(patch,verbose = 0)
    return embedding

def embeddings_comparisons(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0
    euclidean_distance = np.linalg.norm(emb1 - emb2) / len(emb1[0])
    if euclidean_distance < 1:
        return 1 - euclidean_distance
    return 0
def Jaccard_index(d1, d2): #get the jaccard index between the two rectangles d1 and d2
    left = max(d1.bb_left, d2.bb_left)
    right = min(d1.bb_left + d1.bb_width, d2.bb_left + d2.bb_width)
    dx = max(0, right - left)
    top = max(d1.bb_top, d2.bb_top)
    down = min(d1.bb_top + d1.bb_height, d2.bb_top + d2.bb_height)
    dy = max(0, down - top)
    intersect = dx * dy #area of the intersection between d1 and d2
    union = d1.bb_width * d1.bb_height + d2.bb_width * d2.bb_height - intersect #area of the union of d1 and d2
    return intersect / union

def elimination_by_size_ratio(d1, d2):
    area_1 = d1.bb_width * d1.bb_height
    area_2 = d2.bb_width * d2.bb_height
    area_1, area_2 = min(area_1, area_2), max(area_1, area_2)
    if area_2 / area_1 >= 2: #if the biggest box is at least twice as big than the smallest, they can't match
        return 0
    return 1

def similarity_function(d1, d2, emb1, emb2): #get the jaccard_index and the Resnet50 comparison result and cancel it if it is above the minimal similarity value
    jaccard = Jaccard_index(d1, d2)
    distance = embeddings_comparisons(emb1, emb2)
    elimination = elimination_by_size_ratio(d1, d2)
    result = jaccard * 0.9 + distance * 0.1
    if result < 0.1 or distance < 0.8 or elimination == 0:
        return 0
    return result

def get_similarity_matrix(d1, d2, emb1, emb2): #get the similarity matrix between the rectangles of d1 and those of d2
    matrix = np.zeros((d1.shape[0], d2.shape[0]))
    embeddings = []
    for i in range(d1.shape[0]):
        for j in range(d2.shape[0]):
            matrix[i, j] = similarity_function(d1.iloc[i], d2.iloc[j], emb1[i], emb2[j])
    return matrix

def track_matching(matrix): #determines the next organisation of tracks from matrix similarity using linear_sum_assignment function
    new_track = [-1] * len(matrix[0])
    row, col = linear_sum_assignment(matrix, maximize=True)
    for i in range(len(row)):
        if matrix[row[i], col[i]] >= 0.1: #minimal correspondance value
            new_track[col[i]] = row[i]
    return new_track

def get_centroid(box): #get the coordinates of the center of the rectangle box (pair of integers)
    return int(box.bb_left + box.bb_width / 2), int(box.bb_top + box.bb_height / 2)

def get_prediction(box, kalman_filters, id): #predict the next position of the current box through Kalma Filter
    col = list(box.index)
    coord = get_centroid(box)
    update = kalman_filters[id].update([[coord[0]], [coord[1]]])
    prediction = kalman_filters[id].predict()
    result = [box.frame, box.id, box.bb_left + prediction[0][0] - coord[0], box.bb_top + prediction[0][0] - coord[0], box.bb_width, box.bb_height, 1, -1, -1, -1]
    result = pd.DataFrame(result, index=col)
    return result[0]

def track_management(d, old_pos, old_track, new_track, track_history, kalman_filters): #update the current tracks and the tracks history
    track_pos = old_pos.copy()
    for i in range(len(old_track)): #updates already existing tracks, by adding them another position or cancelling them
        id = old_pos.index(i)
        if i in new_track:
            track_pos[id] = new_track.index(i)
            track_history[id].append(get_prediction(d.iloc[track_pos[id]], kalman_filters, id))
        else:
            track_pos[id] = -1
    for i in range(len(new_track)): #creates new tracks
        if new_track[i] == -1:
            track_pos.append(i)
            coord = get_centroid(d.iloc[i])
            kalman_filters.append(KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1, base_x=coord[0], base_y = coord[1]))
            get_prediction(d.iloc[i], kalman_filters, len(kalman_filters) - 1)
            track_history.append([d.iloc[i]])
    return track_pos, track_history, kalman_filters

def get_previous_positions(track_pos, track_history): #get the list of previous boxes positions through track's history
    boxes = []
    for i in range(max(track_pos)):
        id = track_pos.index(i)
        boxes.append(track_history[id][-1])
    return pd.DataFrame(boxes)

def treatment(f, track_pos, old_track, track_history, det, kalman_filters, old_emb): #read picture f, create similarity matrix, throw track management above it and display the results
    img = cv2.imread("ADL-Rundle-6/img1/" + str(f).zfill(6) + ".jpg", cv2.IMREAD_COLOR)
    d1 = get_previous_positions(track_pos, track_history)
    d2 = det[det.frame == f]

    embeddings = []
    for j in range(d2.shape[0]):
        embeddings.append(get_resnet_embedding(img, d2.iloc[j]))
    matrix = get_similarity_matrix(d1, d2, old_emb, embeddings)
    track = track_matching(matrix)
    track_pos, track_history, kalman_filters = track_management(d2, track_pos, old_track, track, track_history, kalman_filters)
    #display predictions for previous tracks
    for i in range(d1.shape[0]):
        box = d1.iloc[i]
        coord = get_centroid(box)
        cv2.rectangle(img, (int(box.bb_left), int(box.bb_top)), (int(box.bb_left) + int(box.bb_width), int(box.bb_top) + int(box.bb_height)), (0, 255, 0), 5)
        cv2.circle(img, coord, 50, (0, 255, 0), 5)
    for i in range(d2.shape[0]):
        id = track_pos.index(i)
        box = track_history[id][-1]
        #display box in picture
        cv2.rectangle(img, (int(box.bb_left), int(box.bb_top)), (int(box.bb_left) + int(box.bb_width), int(box.bb_top) + int(box.bb_height)), (255, 0, 0), 5)
        cv2.putText(img, str(id), (int(box.bb_left), int(box.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        #display history of the box's track
        for j in range(len(track_history[id]) - 1):
            cv2.line(img, get_centroid(track_history[id][j]), get_centroid(track_history[id][j + 1]), (255, 255, 0), 4)
    old_track = track
    cv2.imshow('tracks', img)
    cv2.waitKey(1)
    return track_pos, old_track, track_history, kalman_filters, embeddings

def save_track(det, track_history):  #save the tracks into the file ADL-Rundle-6.txt following the ground truth file format
    count = 0
    for i in track_history:
        count += len(i)
    other = 0
    file = open("ADL-Rundle-6_tp5_resnet_only.txt", 'w')
    min_frame, max_frame = min(det.frame), max(det.frame)
    min_i, max_i = 0, 0
    for f in range(min_frame, max_frame + 1): #for each frame, get the value of the currents tracks positions and write them
        while max_i < len(track_history) and track_history[max_i][0].frame <= f:
            max_i += 1
        for i in range(min_i, max_i):
            index = f - int(track_history[i][0].frame)
            if index < len(track_history[i]):
                box = track_history[i][index]
                file.write(str(f) + ',' + str(i) + ',' + str(box.bb_left) + ',' + str(box.bb_top) + ',' + str(box.bb_width) + ',' + str(box.bb_height) + ',' + str(1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + '\n')
                other += 1
            else:
                if i == min_i:
                    min_i += 1
    file.close()
    return

def TP5():
    #get the first picture
    det = pd.read_csv('ADL-Rundle-6/det/det.txt', names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    min_frame, max_frame = min(det.frame), max(det.frame)
    track_pos, old_track, track_history, kalman_filters = [], [], [], []
    d = det[det.frame == min_frame]
    #initialize the first tracks
    img = cv2.imread("ADL-Rundle-6/img1/" + str(min_frame).zfill(6) + ".jpg", cv2.IMREAD_COLOR)
    embeddings = []
    for i in range(d.shape[0]):
        track_pos.append(i)
        coord = get_centroid(d.iloc[i])
        kalman_filters.append(KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1, base_x=coord[0], base_y = coord[1]))
        get_prediction(d.iloc[i], kalman_filters, len(kalman_filters) - 1)
        track_history.append([d.iloc[i]])
        embeddings.append(get_resnet_embedding(img, d.iloc[i]))
    old_track = track_pos.copy()
    start_time = time.time()
    #continue to get the tracks using previous results over all frames
    for f in range(min_frame + 1, max_frame + 1):
        track_pos, old_track, track_history, kalman_filters, embeddings = treatment(f, track_pos, old_track, track_history, det, kalman_filters, embeddings)
    end_time = time.time()
    print('FPS: ', (end_time - start_time) / (max_frame - min_frame))
    #save the results
    save_track(det, track_history)

TP5()