import cv2
import numpy as np
import pandas as pd
import time

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

def get_similarity_matrix(d1, d2): #get the similarity matrix between the rectangles of d1 and those of d2
    matrix = np.zeros((d1.shape[0], d2.shape[0]))
    for i in range(d1.shape[0]):
        for j in range(d2.shape[0]):
            matrix[i, j] = Jaccard_index(d1.iloc[i], d2.iloc[j])
    return matrix

def track_matching(matrix): #determines the next organisation of tracks from matrix similarity
    max_pos = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    new_track = [-1] * len(matrix[0])
    while matrix[max_pos] >= 0.1: #minimal correspondance value
        matrix[max_pos[0], :] = 0
        matrix[:, max_pos[1]] = 0
        new_track[max_pos[1]] = max_pos[0] + 0
        max_pos = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    return new_track

def get_centroid(box): #get the coordinates of the center of the rectangle box (pair of integers)
    return int(box.bb_left + box.bb_width / 2), int(box.bb_top + box.bb_height / 2)

def track_management(d, old_pos, old_track, new_track, track_history): #update the current tracks and the tracks history
    track_pos = old_pos.copy()
    for i in range(len(old_track)): #updates already existing tracks, by adding them another position or cancelling them
        id = old_pos.index(i)
        if i in new_track:
            track_pos[id] = new_track.index(i)
            track_history[id].append(d.iloc[track_pos[id]])
        else:
            track_pos[id] = -1
    for i in range(len(new_track)): #creates new tracks
        if new_track[i] == -1:
            track_pos.append(i)
            track_history.append([d.iloc[i]])
    return track_pos, track_history

def treatment(f, track_pos, old_track, track_history, det): #read picture f, create similarity matrix, throw track management above it and display the results
    img = cv2.imread("ADL-Rundle-6/img1/" + str(f).zfill(6) + ".jpg", cv2.IMREAD_COLOR)
    d1 = det[det.frame == (f - 1)]
    d2 = det[det.frame == f]
    matrix = get_similarity_matrix(d1, d2)
    track = track_matching(matrix)
    track_pos, track_history = track_management(d2, track_pos, old_track, track, track_history)
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
    return track_pos, old_track, track_history

def TP2():
    #get the first picture
    det = pd.read_csv('ADL-Rundle-6/det/det.txt', names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    min_frame, max_frame = min(det.frame), max(det.frame)
    track_pos, old_track, track_history = [], [], []
    d = det[det.frame == min_frame]
    #initialize the first tracks
    for i in range(d.shape[0]):
        track_pos.append(i)
        track_history.append([d.iloc[i]])
    old_track = track_pos.copy()
    start_time = time.time()
    #continue to get the tracks using previous results over all frames
    for f in range(min_frame + 1, max_frame + 1):
        track_pos, old_track, track_history = treatment(f, track_pos, old_track, track_history, det)
    end_time = time.time()
    print('FPS: ', (end_time - start_time) / (max_frame - min_frame))

TP2()