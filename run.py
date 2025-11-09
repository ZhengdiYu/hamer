from pathlib import Path
import torch
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import cv2
import numpy as np
import pickle

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer
from hamer.utils import recursive_to
from hamer.utils.geometry import aa_to_rotmat, perspective_projection
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
import time

conf_thld_0 = 0.0
conf_thld_1 = 0.0
conf_thld_2 = 0.0
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
FONT_SIZE = 3
FONT_THICKNESS = 1
MARGIN = 10  # pixels
HANDEDNESS_TEXT_COLOR = (205, 54, 88)

# 2D keypoints detection
from vitpose_model import ViTPoseModel
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
# Get the default hand landmarks style
landmark_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
for i in landmark_style.keys():
    landmark_style[i].circle_radius = 1

import json
from typing import Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from typing import List, Tuple
openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
gt_indices = openpose_indices


hand_history = {"left": None, "right": None}
MAX_MOVEMENT = 30  # Maximum allowed pixel movement for hands between frames
handedness_history = []  # Track handedness over time to detect flips

def compute_bbox_from_keypoints(keypoints, img_shape, scale_factor=1.2, max_size_ratio=0.4):
    """
    Compute a tight bounding box from hand keypoints with size constraints.
    
    Args:
        keypoints: (21, 3) array of hand keypoints [x, y, conf]
        img_shape: (H, W) image dimensions
        scale_factor: How much to expand bbox beyond keypoints (default 1.2 = 20% padding)
        max_size_ratio: Maximum bbox size as ratio of image size (default 0.4 = 40%)
    
    Returns:
        bbox: [x_min, y_min, x_max, y_max] format (what ViTDetDataset expects)
    """
    valid = keypoints[:, 2] > 0.3
    if valid.sum() < 3:
        return None
    
    valid_kp = keypoints[valid, :2]
    
    # Get min/max coordinates
    x_min, y_min = valid_kp.min(axis=0)
    x_max, y_max = valid_kp.max(axis=0)
    
    # Compute size
    width = x_max - x_min
    height = y_max - y_min
    
    # Compute center
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    # Add padding
    width *= scale_factor
    height *= scale_factor
    
    # Enforce maximum size (prevent huge boxes)
    img_h, img_w = img_shape[:2]
    max_width = img_w * max_size_ratio
    max_height = img_h * max_size_ratio
    
    if width > max_width:
        print(f"WARNING: Bbox width {width:.0f} exceeds max {max_width:.0f}, clamping")
        width = max_width
    if height > max_height:
        print(f"WARNING: Bbox height {height:.0f} exceeds max {max_height:.0f}, clamping")
        height = max_height
    
    # Make bbox square (use max of width/height)
    size = max(width, height)
    
    # Compute new min/max with padding and size constraints
    new_x_min = cx - size / 2
    new_y_min = cy - size / 2
    new_x_max = cx + size / 2
    new_y_max = cy + size / 2
    
    # Return in [x_min, y_min, x_max, y_max] format (what ViTDetDataset expects)
    return [new_x_min, new_y_min, new_x_max, new_y_max]

def compute_iou(bbox1, bbox2):
    """
    Compute IoU (Intersection over Union) between two bboxes
    bbox format: [x_min, y_min, x_max, y_max]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Compute intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    iou = inter_area / (bbox1_area + bbox2_area - inter_area)
    return iou

def check_bbox_jump(bboxes, is_right, keypoints, last_bbox_dict, img_width, iou_threshold=0.5):
    """
    Detect sudden bbox jumps using IoU (Intersection over Union) with the last frame's bbox.
    If IoU is too low, replace current bbox with last frame's bbox to prevent wrong interpolation.
    
    :param bboxes: list of [x_min, y_min, x_max, y_max]
    :param is_right: list of hand IDs (0=left, 1=right)
    :param keypoints: list of keypoints
    :param last_bbox_dict: dict {hand_id: [x_min, y_min, x_max, y_max]} of last good bboxes
    :param img_width: image width (unused, kept for compatibility)
    :param iou_threshold: IoU threshold below which bbox is considered a jump (e.g., 0.5)
    :return: corrected bboxes, updated last_bbox_dict
    """
    corrected_bboxes = []
    jump_detected = {}
    
    for i, (bbox, hand_id) in enumerate(zip(bboxes, is_right)):
        # Check if we have history for this hand
        if last_bbox_dict[hand_id] is not None:
            last_bbox = last_bbox_dict[hand_id]
            
            # Calculate IoU with last frame's bbox
            iou = compute_iou(bbox, last_bbox)
            
            # If IoU is too low, there's a jump - use last bbox
            if iou < iou_threshold:
                print(f"  ⚠ BBOX JUMP DETECTED for hand {hand_id} ({'right' if hand_id else 'left'}): "
                      f"IoU={iou:.3f} (threshold={iou_threshold})")
                print(f"    Last bbox: [{last_bbox[0]:.0f}, {last_bbox[1]:.0f}, {last_bbox[2]:.0f}, {last_bbox[3]:.0f}]")
                print(f"    New bbox:  [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                print(f"    → Replacing with last frame's bbox")
                
                # Use last frame's bbox instead
                corrected_bboxes.append(last_bbox)
                jump_detected[hand_id] = True
                # Don't update last_bbox_dict - keep the good one
            else:
                # No jump, use current bbox and update history
                corrected_bboxes.append(bbox)
                last_bbox_dict[hand_id] = bbox
                jump_detected[hand_id] = False
        else:
            # First detection, just store it
            corrected_bboxes.append(bbox)
            last_bbox_dict[hand_id] = bbox
            jump_detected[hand_id] = False
    
    return corrected_bboxes, jump_detected


def check_bbox_overlap(bboxes, right, vit_keypoints, iou_threshold=0.3):
    """
    Check if detected left and right hands overlap significantly.
    If they do, keep only the one with higher confidence.
    """
    if len(bboxes) != 2:
        return bboxes, right, vit_keypoints
    
    # Check if we have both left (0) and right (1)
    if not (0 in right and 1 in right):
        return bboxes, right, vit_keypoints
    
    left_idx = np.where(right == 0)[0][0]
    right_idx = np.where(right == 1)[0][0]
    
    iou = compute_iou(bboxes[left_idx], bboxes[right_idx])
    
    if iou > iou_threshold:
        print(f"WARNING: High overlap detected (IoU={iou:.3f}), keeping higher confidence hand")
        
        # Compute confidence for each hand
        left_conf = np.mean(vit_keypoints[left_idx][:, 2])
        right_conf = np.mean(vit_keypoints[right_idx][:, 2])
        
        # Keep the hand with higher confidence
        if left_conf > right_conf:
            return bboxes[[left_idx]], right[[left_idx]], vit_keypoints[[left_idx]]
        else:
            return bboxes[[right_idx]], right[[right_idx]], vit_keypoints[[right_idx]]
    
    return bboxes, right, vit_keypoints

def check_handedness_consistency(bboxes, right, vit_keypoints, history_window=5):
    """
    Check for sudden handedness flips by comparing spatial location with previous frames.
    If a hand suddenly changes handedness but stays in the same spatial location, it's likely a flip.
    
    Args:
        bboxes: (N, 4) array of bounding boxes
        right: (N,) array of handedness (0=left, 1=right)
        vit_keypoints: (N, 21, 3) array of keypoints
        history_window: number of previous frames to check
    
    Returns:
        Corrected bboxes, right, vit_keypoints
    """
    global handedness_history
    
    if len(bboxes) == 0:
        return bboxes, right, vit_keypoints
    
    # SPECIAL CASE: Two hands at same location with different handedness
    # This is almost always a hallucination where one has wrong handedness
    if len(bboxes) == 2 and right[0] != right[1]:
        # Check if they're at the same location
        center_0 = (bboxes[0][:2] + bboxes[0][2:4]) / 2
        center_1 = (bboxes[1][:2] + bboxes[1][2:4]) / 2
        dist = np.linalg.norm(center_0 - center_1)
        
        if dist < 50:  # Very close together (likely same hand)
            print(f"WARNING: Two hands detected at same location (dist={dist:.1f}px) with different handedness!")
            
            # Check history to see which handedness is correct
            if len(handedness_history) >= 1:
                # Look at previous frame to see which handedness was there
                prev_frame = handedness_history[-1]
                
                # Find which hand in previous frame is closest to this location
                avg_center = (center_0 + center_1) / 2
                min_dist = float('inf')
                prev_handedness = None
                
                for prev_hand in prev_frame:
                    d = np.linalg.norm(avg_center - prev_hand['center'])
                    if d < min_dist:
                        min_dist = d
                        prev_handedness = prev_hand['handedness']
                
                if prev_handedness is not None and min_dist < 100:
                    # Keep the hand that matches previous frame's handedness
                    print(f"Previous frame had {'right' if prev_handedness else 'left'} hand at this location. "
                          f"Keeping only that handedness.")
                    
                    if right[0] == prev_handedness:
                        # Keep hand 0
                        return bboxes[[0]], right[[0]], vit_keypoints[[0]]
                    else:
                        # Keep hand 1
                        return bboxes[[1]], right[[1]], vit_keypoints[[1]]
            
            # No history or couldn't determine - keep higher confidence one
            print("No clear history, keeping higher confidence hand")
            conf_0 = np.mean(vit_keypoints[0][:, 2])
            conf_1 = np.mean(vit_keypoints[1][:, 2])
            if conf_0 > conf_1:
                return bboxes[[0]], right[[0]], vit_keypoints[[0]]
            else:
                return bboxes[[1]], right[[1]], vit_keypoints[[1]]
    
    # Store current frame info
    current_frame = []
    for i in range(len(bboxes)):
        bbox_center = (bboxes[i][:2] + bboxes[i][2:4]) / 2
        current_frame.append({
            'bbox': bboxes[i],
            'center': bbox_center,
            'handedness': right[i],
            'keypoints': vit_keypoints[i]
        })
    
    # Check against history if we have enough frames
    if len(handedness_history) >= 2:
        corrected_right = right.copy()
        
        for i, curr_hand in enumerate(current_frame):
            curr_center = curr_hand['center']
            curr_handedness = curr_hand['handedness']
            
            # Look for this hand in previous frames based on spatial proximity
            votes = []  # Collect handedness votes from previous frames
            
            for prev_frame in handedness_history[-history_window:]:
                # Find closest hand in previous frame
                min_dist = float('inf')
                closest_handedness = None
                
                for prev_hand in prev_frame:
                    dist = np.linalg.norm(curr_center - prev_hand['center'])
                    if dist < min_dist and dist < 100:  # Within 100 pixels
                        min_dist = dist
                        closest_handedness = prev_hand['handedness']
                
                if closest_handedness is not None:
                    votes.append(closest_handedness)
            
            # If we have enough votes and they disagree with current detection
            if len(votes) >= 2:
                majority_vote = int(np.median(votes))
                if majority_vote != curr_handedness:
                    print(f"WARNING: Handedness flip detected! "
                          f"Current: {'right' if curr_handedness else 'left'}, "
                          f"History: {'right' if majority_vote else 'left'} "
                          f"(votes: {votes}). Correcting...")
                    corrected_right[i] = majority_vote
        
        # If handedness was corrected, update
        if not np.array_equal(corrected_right, right):
            right = corrected_right
            
            # Check if we now have duplicate handedness (both left or both right)
            # This happens when a hallucination was misclassified with wrong handedness
            if len(right) == 2 and right[0] == right[1]:
                print(f"WARNING: After handedness correction, both hands are {'right' if right[0] else 'left'}. "
                      f"This is likely a hallucination. Keeping only the one with higher confidence.")
                # Keep the one with higher average keypoint confidence
                conf_0 = np.mean(vit_keypoints[0][:, 2])
                conf_1 = np.mean(vit_keypoints[1][:, 2])
                if conf_0 > conf_1:
                    bboxes = bboxes[[0]]
                    right = right[[0]]
                    vit_keypoints = vit_keypoints[[0]]
                else:
                    bboxes = bboxes[[1]]
                    right = right[[1]]
                    vit_keypoints = vit_keypoints[[1]]
    
    # Update history with corrected handedness for next frame
    current_frame_corrected = []
    for i in range(len(bboxes)):
        bbox_center = (bboxes[i][:2] + bboxes[i][2:4]) / 2
        current_frame_corrected.append({
            'bbox': bboxes[i],
            'center': bbox_center,
            'handedness': right[i],
            'keypoints': vit_keypoints[i]
        })
    
    # Add current frame to history
    handedness_history.append(current_frame_corrected)
    
    # Keep only recent history
    if len(handedness_history) > history_window:
        handedness_history.pop(0)
    
    return bboxes, right, vit_keypoints

def restrict_hand_movement(bboxes, right, vit_keypoints):
    global hand_history
    
    if len(bboxes) == 1:
        # Only one hand detected, update history accordingly
        if right[0] == 0:
            hand_history["left"] = (bboxes[0], vit_keypoints[0])
        else:
            hand_history["right"] = (bboxes[0], vit_keypoints[0])
        return bboxes, right, vit_keypoints
    
    # Two hands detected
    left_hand, right_hand = None, None
    left_hand_kp, right_hand_kp = None, None
    
    for i in range(2):
        if right[i] == 0:
            left_hand, left_hand_kp = bboxes[i], vit_keypoints[i]
        else:
            right_hand, right_hand_kp = bboxes[i], vit_keypoints[i]
    
    if hand_history["left"] is not None and left_hand is not None:
        prev_left_bbox, prev_left_kp = hand_history["left"]
        movement = np.linalg.norm(prev_left_bbox[:2] - left_hand[:2])
        if movement > MAX_MOVEMENT:
            left_hand, left_hand_kp = prev_left_bbox, prev_left_kp  # Use last frame's data
    
    if hand_history["right"] is not None and right_hand is not None:
        prev_right_bbox, prev_right_kp = hand_history["right"]
        movement = np.linalg.norm(prev_right_bbox[:2] - right_hand[:2])
        if movement > MAX_MOVEMENT:
            right_hand, right_hand_kp = prev_right_bbox, prev_right_kp  # Use last frame's data
    
    # Update history
    hand_history["left"] = (left_hand, left_hand_kp)
    hand_history["right"] = (right_hand, right_hand_kp)
    
    # Rebuild final lists with filtered results
    new_bboxes, new_right, new_vit_keypoints = [], [], []
    if left_hand is not None:
        new_bboxes.append(left_hand)
        new_right.append(0)
        new_vit_keypoints.append(left_hand_kp)
    if right_hand is not None:
        new_bboxes.append(right_hand)
        new_right.append(1)
        new_vit_keypoints.append(right_hand_kp)
    
    return np.array(new_bboxes), np.array(new_right), np.array(new_vit_keypoints)


def run_mediapipe(img, vis):
    # mediapipe 啟用偵測手掌
    with mp_hands.Hands(
        model_complexity=1,
        # max_num_hands=1,
        min_detection_confidence=conf_thld_0,
        min_tracking_confidence=conf_thld_0) as hands:

        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)

        # Draw the hand annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        #print('running old mediapipe v1!')
        left_hand = None
        right_hand = None
        #print('*******')
        if results.multi_hand_landmarks:
            for hand_landmarks, classification in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness = classification.classification[0].label

                lmList = []
                # confidences = [landmark.presence for landmark in hand_landmarks.landmark]
                for _, lm in enumerate(hand_landmarks.landmark):
                    if lm.visibility != 0:
                        raise ValueError
                    h, w, c = img.shape
                    px, py = int(lm.x * w), int(lm.y * h)
                    # lmList.append([px, py, lm.presence])
                    lmList.append([px, py, 1])

                # vis
                mp_drawing.draw_landmarks(
                    vis,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_style,
                    mp_drawing_styles.get_default_hand_connections_style())

                if handedness == 'Left':
                    cv2.putText(vis, f"Right",
                    (0, 40), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                    right_hand = np.array(lmList)

                elif handedness == 'Right':
                    cv2.putText(vis, f"Left",
                    (0, 80), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                    left_hand = np.array(lmList)
                else:
                    raise ValueError

        return left_hand, right_hand, vis

def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    # import IPython; IPython.embed(); exit()
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]

    # unnormalize to crop coords
    # keypoints = 0.5 * crop_size * (keypoints + 1.0)

    # rescale to orig img crop
    keypoints *= h[..., None, None] / crop_size

    # transform into original image coords
    keypoints[:,:,0] = (cx - h/2)[..., None] + keypoints[:,:,0]
    keypoints[:,:,1] = (cy - h/2)[..., None] + keypoints[:,:,1]
    return keypoints

def get_keypoints_rectangle(keypoints: np.array, threshold: float) -> Tuple[float, float, float]:
    """
    Compute rectangle enclosing keypoints above the threshold.
    Args:
        keypoints (np.array): Keypoint array of shape (N, 3).
        threshold (float): Confidence visualization threshold.
    Returns:
        Tuple[float, float, float]: Rectangle width, height and area.
    """
    print(keypoints.shape)
    valid_ind = keypoints[:, -1] > threshold
    if valid_ind.sum() > 0:
        valid_keypoints = keypoints[valid_ind][:, :-1]
        max_x = valid_keypoints[:,0].max()
        max_y = valid_keypoints[:,1].max()
        min_x = valid_keypoints[:,0].min()
        min_y = valid_keypoints[:,1].min()
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        return width, height, area
    else:
        return 0,0,0

def render_keypoints(img: np.array,
                     keypoints: np.array,
                     pairs: List,
                     colors: List,
                     thickness_circle_ratio: float,
                     thickness_line_ratio_wrt_circle: float,
                     pose_scales: List,
                     threshold: float = 0.1,
                     alpha: float = 1.0) -> np.array:
    """
    Render keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        keypoints (np.array): Keypoint array of shape (N, 3).
        pairs (List): List of keypoint pairs per limb.
        colors: (List): List of colors per keypoint.
        thickness_circle_ratio (float): Circle thickness ratio.
        thickness_line_ratio_wrt_circle (float): Line thickness ratio wrt the circle.
        pose_scales (List): List of pose scales.
        threshold (float): Only visualize keypoints with confidence above the threshold.
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """
    img_orig = img.copy()
    width, height = img.shape[1], img.shape[2]
    area = width * height

    lineType = 8
    shift = 0
    numberColors = len(colors)
    thresholdRectangle = 0.1

    person_width, person_height, person_area = get_keypoints_rectangle(keypoints, thresholdRectangle)
    if person_area > 0:
        ratioAreas = min(1, max(person_width / width, person_height / height))
        thicknessRatio = np.maximum(np.round(math.sqrt(area) * thickness_circle_ratio * ratioAreas), 2)
        thicknessCircle = np.maximum(1, thicknessRatio if ratioAreas > 0.05 else -np.ones_like(thicknessRatio))
        thicknessLine = np.maximum(1, np.round(thicknessRatio * thickness_line_ratio_wrt_circle))
        radius = thicknessRatio / 2

        img = np.ascontiguousarray(img.copy())
        for i, pair in enumerate(pairs):
            index1, index2 = pair
            if keypoints[index1, -1] > threshold and keypoints[index2, -1] > threshold:
                thicknessLineScaled = int(round(min(thicknessLine[index1], thicknessLine[index2]) * pose_scales[0]))
                colorIndex = index2
                color = colors[colorIndex % numberColors]
                keypoint1 = keypoints[index1, :-1].astype(np.int32)
                keypoint2 = keypoints[index2, :-1].astype(np.int32)
                cv2.line(img, tuple(keypoint1.tolist()), tuple(keypoint2.tolist()), tuple(color.tolist()), thicknessLineScaled, lineType, shift)
        for part in range(len(keypoints)):
            faceIndex = part
            if keypoints[faceIndex, -1] > threshold:
                radiusScaled = int(round(radius[faceIndex] * pose_scales[0]))
                thicknessCircleScaled = int(round(thicknessCircle[faceIndex] * pose_scales[0]))
                colorIndex = part
                color = colors[colorIndex % numberColors]
                center = keypoints[faceIndex, :-1].astype(np.int32)
                cv2.circle(img, tuple(center.tolist()), radiusScaled, tuple(color.tolist()), thicknessCircleScaled, lineType, shift)
    return img

def render_hand_keypoints(img, right_hand_keypoints, threshold=0.1, use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0):
    if use_confidence and map_fn is not None:
        #thicknessCircleRatioLeft = 1./50 * map_fn(left_hand_keypoints[:, -1])
        thicknessCircleRatioRight = 1./50 * map_fn(right_hand_keypoints[:, -1])
    else:
        #thicknessCircleRatioLeft = 1./50 * np.ones(left_hand_keypoints.shape[0])
        thicknessCircleRatioRight = 1./50 * np.ones(right_hand_keypoints.shape[0])
    thicknessLineRatioWRTCircle = 0.75
    pairs = [0,1,  1,2,  2,3,  3,4,  0,5,  5,6,  6,7,  7,8,  0,9,  9,10,  10,11,  11,12,  0,13,  13,14,  14,15,  15,16,  0,17,  17,18,  18,19,  19,20]
    pairs = np.array(pairs).reshape(-1,2)

    colors = [100.,  100.,  100.,
              100.,    0.,    0.,
              150.,    0.,    0.,
              200.,    0.,    0.,
              255.,    0.,    0.,
              100.,  100.,    0.,
              150.,  150.,    0.,
              200.,  200.,    0.,
              255.,  255.,    0.,
                0.,  100.,   50.,
                0.,  150.,   75.,
                0.,  200.,  100.,
                0.,  255.,  125.,
                0.,   50.,  100.,
                0.,   75.,  150.,
                0.,  100.,  200.,
                0.,  125.,  255.,
              100.,    0.,  100.,
              150.,    0.,  150.,
              200.,    0.,  200.,
              255.,    0.,  255.]
    colors = np.array(colors).reshape(-1,3)
    #colors = np.zeros_like(colors)
    poseScales = [1]
    #img = render_keypoints(img, left_hand_keypoints, pairs, colors, thicknessCircleRatioLeft, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    img = render_keypoints(img, right_hand_keypoints, pairs, colors, thicknessCircleRatioRight, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    #img = render_keypoints(img, right_hand_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1)
    return img

def render_body_keypoints(img: np.array,
                          body_keypoints: np.array) -> np.array:
    """
    Render OpenPose body keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """

    thickness_circle_ratio = 1./75. * np.ones(body_keypoints.shape[0])
    thickness_line_ratio_wrt_circle = 0.75
    pairs = []
    pairs = [1,8,1,2,1,5,2,3,3,4,5,6,6,7,8,9,9,10,10,11,8,12,12,13,13,14,1,0,0,15,15,17,0,16,16,18,14,19,19,20,14,21,11,22,22,23,11,24]
    pairs = np.array(pairs).reshape(-1,2)
    colors = [255.,     0.,     85.,
              255.,     0.,     0.,
              255.,    85.,     0.,
              255.,   170.,     0.,
              255.,   255.,     0.,
              170.,   255.,     0.,
               85.,   255.,     0.,
                0.,   255.,     0.,
              255.,     0.,     0.,
                0.,   255.,    85.,
                0.,   255.,   170.,
                0.,   255.,   255.,
                0.,   170.,   255.,
                0.,    85.,   255.,
                0.,     0.,   255.,
              255.,     0.,   170.,
              170.,     0.,   255.,
              255.,     0.,   255.,
               85.,     0.,   255.,
                0.,     0.,   255.,
                0.,     0.,   255.,
                0.,     0.,   255.,
                0.,   255.,   255.,
                0.,   255.,   255.,
                0.,   255.,   255.]
    colors = np.array(colors).reshape(-1,3)
    pose_scales = [1]
    return render_keypoints(img, body_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1)

def render_openpose(img: np.array,
                    hand_keypoints: np.array) -> np.array:
    """
    Render keypoints in the OpenPose format on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """
    #img = render_body_keypoints(img, body_keypoints)
    img = render_hand_keypoints(img, hand_keypoints)
    return img

def save_video(path, out_dir, out_name):
    print('saving to :', out_name + '.mp4')
    img_array = []
    height, width = 0, 0
    for filename in tqdm(sorted(os.listdir(path))):
    # for filename in tqdm(sorted(os.listdir(path), key=lambda x:int(os.path.basename(x).split('.')[0]))):
        img = cv2.imread(path + '/' + filename)
        if height != 0:
            img = cv2.resize(img, (width, height))
        height, width, _ = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(os.path.join(out_dir, out_name + '.mp4'), 0x7634706d, 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('done')

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--res_folder', type=str, help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=1.3, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--conf', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--type', type=str, default='EgoDexter', help='Path to pretrained model checkpoint')
    parser.add_argument('--render', dest='render', action='store_true', default=False, help='If set, render side view also')
    
    args = parser.parse_args()

    # Download and load checkpoints
    CACHE_DIR_HAMER = args.checkpoint
    print("args.checkpoint: ", args.checkpoint, CACHE_DIR_HAMER)
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(parent_path=CACHE_DIR_HAMER, device=device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    args.out_folder = args.out_folder + '_' +str(model_cfg.EXTRA.FOCAL_LENGTH)
    # os.makedirs(args.out_folder, exist_ok=True)
    render_save_path = os.path.join(os.path.dirname(args.res_folder), f'render_all_{model_cfg.EXTRA.FOCAL_LENGTH}')
    # depth_save_path = os.path.join(os.path.dirname(args.res_folder), 'depth_all')
    joint2d_save_path = os.path.join(os.path.dirname(args.res_folder), f'joint2d_{model_cfg.EXTRA.FOCAL_LENGTH}')
    vit_save_path = os.path.join(os.path.dirname(args.res_folder), f'vit_{model_cfg.EXTRA.FOCAL_LENGTH}')
    mesh_dir = os.path.join(os.path.dirname(args.res_folder), f'mesh_{model_cfg.EXTRA.FOCAL_LENGTH}')
    os.makedirs(render_save_path, exist_ok=True)
    os.makedirs(joint2d_save_path, exist_ok=True)
    os.makedirs(vit_save_path, exist_ok=True)
    # os.makedirs(depth_save_path, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.res_folder), exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]

    # Iterate over all images in folder
    img_list = []
    print('start iteration.')
    results_dict = {}

    big_all_verts = []
    big_all_cam_t = []
    big_all_joints = []
    big_all_right = []

    tid = []
    x= 0
    tracked_time = [0,0]
    
    # Track bbox history to detect sudden jumps using IoU
    last_bbox_dict = {0: None, 1: None}  # {hand_id: [x_min, y_min, x_max, y_max]}

    for img_path in tqdm(sorted(img_paths)):
        a = time.time()
        img_path = str(img_path)
        img_cv2 = cv2.imread(str(img_path))

        results_dict[img_path] = {}
        results_dict[img_path]['mano'] = []
        results_dict[img_path]['cam_trans'] = []
        results_dict[img_path]['tracked_ids'] = []
        results_dict[img_path]['tracked_time'] = []
        results_dict[img_path]['extra_data'] = []

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # mediapipe
        # vis = img_cv2.copy()
        # left, right, vis = run_mediapipe(img_cv2.copy(), vis)
        # print(left, right, vis.shape)
        # cv2.imwrite(f'./{x}.jpg', vis)
        # x += 1

        bboxes = []
        is_right = []
        vit_keypoints_list = []
        conf_list = []

        # Use hands based on hand keypoint detections
        last_left_conf = None
        last_right_conf = None
        l_flag = False
        r_flag = False
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        # conf_list = []
        # for index, vitposes in enumerate(vitposes_out):
        #     left_conf = np.mean(vitposes['keypoints'][-42:-21][:,2]) * sum(vitposes['keypoints'][-42:-21][:,2] > 0.5)
        #     right_conf = np.mean(vitposes['keypoints'][-21:][:,2]) * sum(vitposes['keypoints'][-21:][:,2] > 0.5)
        #     print(np.mean(vitposes['keypoints'][-42:-21][:,2]), sum(vitposes['keypoints'][-42:-21][:,2] > 0.5))
        #     print(np.mean(vitposes['keypoints'][-21:][:,2]), sum(vitposes['keypoints'][-21:][:,2] > 0.5))
        #     total_conf = left_conf + right_conf
        #     print(total_conf, left_conf, right_conf)
        #     print('*-****')
        #     conf_list.append(total_conf)

        # vitposes_out = [vitposes_out[np.argsort(conf_list)[-1]]]
        # print(conf_list, np.argsort(conf_list), 'selected: ', conf_list[np.argsort(conf_list)[-1]])

        X = 0
        # # option 2.
        # for index, vitposes in enumerate(vitposes_out):
        #     # print(vitposes.keys()) # dict_keys(['bbox', 'keypoints'])
        #     left_hand_keyp = vitposes['keypoints'][-42:-21]
        #     right_hand_keyp = vitposes['keypoints'][-21:]

        #     # Rejecting not confident detections
        #     keyp = left_hand_keyp
        #     valid = keyp[:,2] > 0.5
        #     if sum(valid) > 5:
        #         print('append left')
        #         bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
        #         # bboxes.append(bbox)
        #         is_right.append(0)
        #         # vit_keypoints_list.append(keyp)
        #         l_bbox = bbox
        #         l_keyp = keyp
        #         last_left_conf = np.mean(keyp[:,2])
        #         l_flag = True

        #     keyp = right_hand_keyp
        #     valid = keyp[:,2] > 0.5
        #     if sum(valid) > 5:
        #         print('append right')
        #         bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
        #         # bboxes.append(bbox)
        #         is_right.append(1)
        #         # vit_keypoints_list.append(keyp)
        #         r_bbox = bbox
        #         r_keyp = keyp
        #         last_right_conf = np.mean(keyp[:,2])
        #         r_flag = True

        #     X += 1
        # assert X == 1

        # option 1.
        print("person number: ", len(vitposes_out))
        for index, vitposes in enumerate(vitposes_out):
            # print(vitposes.keys()) # dict_keys(['bbox', 'keypoints'])
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            # all_vit_2d = np.stack((left_hand_keyp, right_hand_keyp))
            # print(all_vit_2d.shape)
            # vit_img = img_cv2.copy()[:,:,::-1] * 255
            # for i in range(len(all_vit_2d)):
            #     body_keypoints_2d = all_vit_2d[i, :21].copy()
            #     for op, gt in zip(openpose_indices, gt_indices):
            #         if all_vit_2d[i, gt, -1] > body_keypoints_2d[op, -1]:
            #             body_keypoints_2d[op] = all_vit_2d[i, gt]
            #     vit_img = render_openpose(vit_img, body_keypoints_2d)
            #     vit_img = vit_img.astype(np.uint8)

            # cv2.imwrite(f'{img_fn}_{index}.jpg', vit_img[:, :, ::-1])

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.3
            print("left valid: ", valid, sum(valid), np.mean(keyp[:,2]), last_left_conf)
            if sum(valid) > 10:
                if 0 not in is_right:
                    print('append left')
                    bbox = compute_bbox_from_keypoints(keyp, img_cv2.shape, scale_factor=1.5, max_size_ratio=0.4)
                    if bbox is None:
                        continue
                    # bboxes.append(bbox)
                    is_right.append(0)
                    # vit_keypoints_list.append(keyp)
                    l_bbox = bbox
                    l_keyp = keyp
                    last_left_conf = np.mean(keyp[:,2])
                    l_flag = True
                else:
                    if np.mean(keyp[:,2]) > last_left_conf:
                        print('exchange!!!!!: ', np.mean(keyp[:,2]), last_left_conf)
                        # raise ValueError
                        bbox = compute_bbox_from_keypoints(keyp, img_cv2.shape, scale_factor=1.5, max_size_ratio=0.4)
                        if bbox is not None:
                            last_left_conf = np.mean(keyp[:,2])
                            l_bbox = bbox
                            l_keyp = keyp

            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.3
            print("right valid: ", valid, sum(valid),np.mean(keyp[:,2]), last_right_conf)
            if sum(valid) > 10:
                if 1 not in is_right:
                    print('append right')
                    bbox = compute_bbox_from_keypoints(keyp, img_cv2.shape, scale_factor=1.5, max_size_ratio=0.4)
                    if bbox is None:
                        continue
                    # bboxes.append(bbox)
                    is_right.append(1)
                    # vit_keypoints_list.append(keyp)
                    r_bbox = bbox
                    r_keyp = keyp
                    last_right_conf = np.mean(keyp[:,2])
                    r_flag = True
                else:
                    if np.mean(keyp[:,2]) > last_right_conf:
                        print('exchange!!!!!: ', np.mean(keyp[:,2]), last_right_conf)
                        # raise ValueError
                        bbox = compute_bbox_from_keypoints(keyp, img_cv2.shape, scale_factor=1.5, max_size_ratio=0.4)
                        if bbox is not None:
                            last_right_conf = np.mean(keyp[:,2])
                            r_bbox = bbox
                            r_keyp = keyp

        is_right = []
        if l_flag:
            bboxes.append(l_bbox)
            is_right.append(0)
            vit_keypoints_list.append(l_keyp)

        if r_flag:
            bboxes.append(r_bbox)
            is_right.append(1)
            vit_keypoints_list.append(r_keyp)

        if len(bboxes) == 0:
            results_dict[img_path]['tid'] = tid
            results_dict[img_path]['tracked_time'] = []
            results_dict[img_path]['shot'] = 0
            tracked_time[0] += 1
            tracked_time[1] += 1
            for i in tid:
                results_dict[img_path]['tracked_time'].append(tracked_time[i])
            for idx, i in enumerate(tracked_time):
                if i > 50 and (idx in tid):
                    tid.remove(idx)
            print('no hand detected!!!', results_dict[img_path]['tid'], results_dict[img_path]['tracked_ids'], results_dict[img_path]['tracked_time'])
            if args.render:
                print('save no hand detected image')
                cv2.imwrite(os.path.join(render_save_path, f'{img_fn}.jpg'), img_cv2.astype(np.float32))
                cv2.imwrite(os.path.join(joint2d_save_path, f'{img_fn}.jpg'), img_cv2.astype(np.float32))
                cv2.imwrite(os.path.join(vit_save_path, f'{img_fn}.jpg'), img_cv2.astype(np.float32))
            continue

        if len(bboxes) > 0:
            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            vit_keypoints = np.stack(vit_keypoints_list)

            sort_idx = np.argsort(right)
            print('sort_idx', sort_idx, right)
            right = right[sort_idx][:2]
            print('right: ', right)
            vit_keypoints = vit_keypoints[sort_idx][:2]
            boxes = boxes[sort_idx][:2]
            
            print(f"Before handedness check: {len(boxes)} hands, handedness: {right}")
            
            # Visualize bboxes before any filtering
            img_debug = img_cv2.copy()
            for i, bbox in enumerate(boxes):
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color = (0, 255, 0) if right[i] == 1 else (255, 0, 0)  # Green=right, Blue=left
                cv2.rectangle(img_debug, (x1, y1), (x2, y2), color, 3)
                label = f"{'R' if right[i] == 1 else 'L'}"
                cv2.putText(img_debug, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imwrite(os.path.join(render_save_path, f'{img_fn}_bbox_before.jpg'), img_debug)
            
            # DISABLE handedness check for now to debug
            # Check for handedness flips FIRST (temporal consistency)
            # This must run before overlap check to ensure we keep the correct handedness
            # boxes, right, vit_keypoints = check_handedness_consistency(boxes, right, vit_keypoints, history_window=5)
            
            print(f"After handedness check (DISABLED): {len(boxes)} hands, handedness: {right}")
            
            # Check for overlapping hands (hallucination detection) FIRST
            # Remove duplicates before any tracking
            boxes, right, vit_keypoints = check_bbox_overlap(boxes, right, vit_keypoints, iou_threshold=0.3)
            
            print(f"After overlap check: {len(boxes)} hands, handedness: {right}")
            
            # NOW check for bbox jumps and update tracking with VALID hands only
            # This ensures we only track bboxes that survived filtering
            img_width = img_cv2.shape[1]
            print(f"\nFrame {img_fn}: Before jump check - boxes shape: {boxes.shape if isinstance(boxes, np.ndarray) else len(boxes)}")
            for i, (bbox, hand_id) in enumerate(zip(boxes, right)):
                if last_bbox_dict[hand_id] is not None:
                    iou = compute_iou(bbox, last_bbox_dict[hand_id])
                    print(f"  Hand {hand_id} ({'R' if hand_id else 'L'}): IoU with last frame = {iou:.3f}")
                else:
                    print(f"  Hand {hand_id} ({'R' if hand_id else 'L'}): First detection (no history)")
            
            boxes, jump_detected = check_bbox_jump(boxes, right, vit_keypoints, last_bbox_dict, img_width, iou_threshold=0.3)
            boxes = np.array(boxes)  # Convert back to numpy array
            
            print(f"After bbox jump check: {len(boxes)} hands, handedness: {right}")
            
            # Visualize bboxes after filtering
            img_debug = img_cv2.copy()
            for i, bbox in enumerate(boxes):
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color = (0, 255, 0) if right[i] == 1 else (255, 0, 0)  # Green=right, Blue=left
                cv2.rectangle(img_debug, (x1, y1), (x2, y2), color, 3)
                label = f"{'R' if right[i] == 1 else 'L'}"
                cv2.putText(img_debug, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imwrite(os.path.join(render_save_path, f'{img_fn}_bbox_after.jpg'), img_debug)
        else:
            raise ValueError

        # Run reconstruction on all detected hands
        assert (right == [0,1]).all() or (right == [0]).all() or (right == [1]).all()

        ####################################

        # print(f"Before restrict_hand_movement: {len(boxes)} hands, handedness: {right}")
        # boxes, right, vit_keypoints = restrict_hand_movement(boxes, right, vit_keypoints)
        # print(f"After restrict_hand_movement: {len(boxes)} hands, handedness: {right}")

        if len(boxes) == 0:
            print("WARNING: All hands filtered out by restrict_hand_movement!")

        print(f"Creating dataset with rescale_factor={args.rescale_factor}")
        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            patch_size = args.rescale_factor * max(width, height)
            print(f"  Hand {i} ({'right' if right[i] else 'left'}): bbox size={width:.0f}x{height:.0f}, patch size={patch_size:.0f}")
        
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, vit_keypoints, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_joints = []
        all_pred_3d = []
        all_right = []
        all_vit_2d = []
        all_pred_2d = []
        all_bboxes = []

        left_flag = False
        right_flag = False
        print(f"Starting HaMeR reconstruction with {len(boxes)} hands")
        b = time.time()
        print('vit time:', b-a)
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            
            # Compute cam_trans using SLAHMR-style robust method
            # This ensures depth (tz) is always positive and computed from bbox scale
            batch_size = batch['img'].shape[0]
            pred_cam_t_full = torch.zeros(batch_size, 3, device=pred_cam.device)
            for n in range(batch_size):
                cam = pred_cam[n]  # (3,) - [scale, tx, ty] in normalized coords
                H, W = img_size[n, 1], img_size[n, 0]  # height, width
                focal = scaled_focal_length[n] if scaled_focal_length.ndim > 0 else scaled_focal_length
                
                cx, cy = box_center[n]  # bbox center
                scale = box_size[n]  # bbox size
                
                # Compute depth from bbox scale (ensures positive depth)
                tz = 2 * focal / (scale * cam[0] + 1e-6)
                # Adjust translation based on bbox center offset from image center
                tx = cam[1] + tz / focal * (cx - W / 2)
                ty = cam[2] + tz / focal * (cy - H / 2)
                
                pred_cam_t_full[n] = torch.tensor([tx, ty, tz], device=pred_cam.device)
            
            print(f"[CAM_TRANS] Computed using SLAHMR method - Z range: [{pred_cam_t_full[:, 2].min():.3f}, {pred_cam_t_full[:, 2].max():.3f}]")
            # d3 = out['pred_keypoints_3d'].reshape(batch_size, -1, 3)
            # out['pred_keypoints_2d'] = perspective_projection(d3,
            #                             translation=pred_cam_t_full.reshape(batch_size, 3),
            #                             focal_length=out['focal_length'].reshape(-1, 2) / model_cfg.MODEL.IMAGE_SIZE)

            pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()

            # print('batch_size: ', batch_size)
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()
                
                # Visualize the input patch fed to HaMeR
                is_right = int(batch['right'][n].cpu().numpy())
                patch_vis = (input_patch * 255).astype(np.uint8)
                hand_label = 'right' if is_right == 1 else 'left'
                patch_path = os.path.join(render_save_path, f'{img_fn}_patch_{hand_label}.jpg')
                cv2.imwrite(patch_path, patch_vis[:, :, ::-1])  # Convert RGB to BGR for cv2

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                pred_joints = out['pred_keypoints_2d'][n].detach().cpu().numpy()
                pred_3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()

                is_right = int(batch['right'][n].cpu().numpy())

                if is_right == 1:
                    if not right_flag:
                        right_flag = True
                        print(f"  → Adding RIGHT hand (first one)")
                    else:
                        print(f"  → Skipping RIGHT hand (already have one)")
                        continue
                else:
                    if not left_flag:
                        left_flag = True
                        print(f"  → Adding LEFT hand (first one)")
                    else:
                        print(f"  → Skipping LEFT hand (already have one)")
                        continue

                pred_joints[:,0] = (2*is_right-1)*pred_joints[:,0]
                v = np.ones((21, 1))
                pred_joints = np.concatenate((pred_joints, v), axis=-1)
                verts[:,0] = (2*is_right-1)*verts[:,0]
                pred_3d[:,0] = (2*is_right-1)*pred_3d[:,0]
                cam_t = pred_cam_t_full[n]
                # print('cam_t.shape', cam_t.shape)

                all_pred_3d.append(pred_3d)
                all_joints.append(pred_joints)
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_pred_2d.append(pred_joints)
                all_vit_2d.append(batch['2d'][n])
                all_bboxes.append(batch['bbox'][n].detach().cpu().numpy())

                tracked_time[is_right] = 0
                results_dict[img_path]['tracked_ids'].append(is_right)
                if is_right not in tid:
                    tid.append(is_right)

                out['pred_mano_params'][n]['is_right'] = is_right
                results_dict[img_path]['mano'].append(out['pred_mano_params'][n])
                results_dict[img_path]['cam_trans'].append(cam_t)
                # if len(all_joints) == 180:
                #     break

        big_all_joints.append(all_pred_3d)
        big_all_verts.append(all_verts)
        big_all_cam_t.append(all_cam_t)
        big_all_right.append(all_right)

        print(f"Finished HaMeR reconstruction: {len(all_verts)} hands reconstructed")
        if len(all_verts) == 0:
            print("WARNING: No hands were reconstructed by HaMeR!")

        assert len(results_dict[img_path]['tracked_ids']) <= 2
        if len(results_dict[img_path]['tracked_ids']) == 1:
            if results_dict[img_path]['tracked_ids'][0] == 0: # if is_right == 0, left hand presents
                tracked_time[1] += 1 # then right hand time + 1
            else:
                tracked_time[0] += 1

        tid = sorted(tid)
        for idx, i in enumerate(tracked_time):
            if i > 50 and (idx in tid):
                tid.remove(idx)

        results_dict[img_path]['shot'] = 0
        results_dict[img_path]['tracked_time'] = []
        for i in tid:
            results_dict[img_path]['tracked_time'].append(tracked_time[i])

        results_dict[img_path]['tid'] = np.array(tid)
        print('tid/tracked_ids/tracked_time', results_dict[img_path]['tid'], results_dict[img_path]['tracked_ids'], results_dict[img_path]['tracked_time'])

        if args.full_frame and len(all_verts) > 0:
            # Render front view
            assert len(all_vit_2d) == len(all_pred_2d)
            assert len(all_verts) == 2 or len(all_verts) == 1, f'{len(all_verts)}'
            print('length: ', len(all_vit_2d), len(all_pred_2d))
            all_vit_2d = torch.stack(all_vit_2d).cpu().numpy()
            all_pred_2d = np.stack(all_pred_2d)
            all_bboxes = np.stack(all_bboxes)
            # print('vit_2d: ', all_vit_2d.shape)
            print('pred_2d: ', all_pred_2d.shape)
            print('all_bboxes: ', all_bboxes.shape)
            all_pred_2d = model_cfg.MODEL.IMAGE_SIZE * (all_pred_2d + 0.5)
            all_pred_2d = convert_crop_coords_to_orig_img(bbox=all_bboxes, keypoints=all_pred_2d, crop_size=model_cfg.MODEL.IMAGE_SIZE)
            assert len(all_pred_2d.shape) == 3
            assert all_pred_2d.shape[-1] == 3
            all_pred_2d[:,:,-1] = 1
            results_dict[img_path]['extra_data'] = []
            for i in range(len(all_pred_2d)):
                results_dict[img_path]['extra_data'].append(all_pred_2d[i].tolist())

            if args.render:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                # print(all_cam_t)
                cam_view, multi_depth = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

                # Overlay image
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                # Apply colormap to project depth into color
                # alpha_channel = multi_depth.copy()
                # colormap = cm.get_cmap('gray')
                # multi_depth = (colormap(multi_depth)[:, :, :3] * 255).astype(np.uint8)
                # multi_depth = np.dstack([multi_depth, (alpha_channel * 255).astype(np.uint8)])

                # # save to .jpg
                # plt.imsave('depth_visualization.jpg', depth_colored)
                # depth_img_overlay = input_img[:,:,:3] * (1-multi_depth[:,:,3:]) + multi_depth[:,:,:3] * multi_depth[:,:,3:]

                vit_img = input_img.copy()[:,:,::-1] * 255
                for i in range(len(all_verts)):
                    body_keypoints_2d = all_vit_2d[i, :21].copy()
                    for op, gt in zip(openpose_indices, gt_indices):
                        if all_vit_2d[i, gt, -1] > body_keypoints_2d[op, -1]:
                            body_keypoints_2d[op] = all_vit_2d[i, gt]
                    vit_img = render_openpose(vit_img, body_keypoints_2d)
                    cx, cy, h, w = all_bboxes[i]
                    print('bbox: ', cx, cy, h, w, vit_img.shape)

                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)

                    # print(pred_img.shape)
                    vit_img = vit_img.astype(np.uint8)
                    cv2.rectangle(vit_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # pred_img = batch['img_patch'][n].cpu().numpy().copy() # input_img.copy()[:,:,::-1] * 255
                pred_img = input_img.copy()[:,:,:-1][:,:,::-1] * 255
                for i in range(len(all_verts)):
                    body_keypoints_2d = all_pred_2d[i, :21].copy()
                    for op, gt in zip(openpose_indices, gt_indices):
                        if all_pred_2d[i, gt, -1] > body_keypoints_2d[op, -1]:
                            body_keypoints_2d[op] = all_pred_2d[i, gt]
                            raise ValueError
                        else:
                            pass

                    assert (body_keypoints_2d == all_pred_2d[i, :21]).all()
                    pred_img = render_openpose(pred_img, body_keypoints_2d)

                # draw 2d keypoints
                cv2.imwrite(os.path.join(render_save_path, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])
                cv2.imwrite(os.path.join(joint2d_save_path, f'{img_fn}.jpg'), pred_img)
                cv2.imwrite(os.path.join(vit_save_path, f'{img_fn}.jpg'), vit_img)

        c = time.time()
        print('one step time: ', c - a, 'hamer time: ', c-b)

    # Save all meshes to disk
    # print(big_all_cam_t)
    # print(len(big_all_cam_t), len(big_all_cam_t[0]), len(big_all_cam_t[0][0]))
    if len(big_all_joints[0]) == 1:
        init_trans = big_all_cam_t[0][0].copy() + big_all_joints[0][0][9]
    else:
        x = (big_all_cam_t[0][0] + big_all_joints[0][0][9] + big_all_cam_t[0][1] + big_all_joints[0][1][9]) / 2
        init_trans = big_all_cam_t[0][0].copy()

    # print(big_all_cam_t[0])
    N = 0
    for a, b, c, d in zip(big_all_verts, big_all_joints, big_all_cam_t, big_all_right):
        for verts, joints, cam_t, is_right in zip(a, b, c, d):
            if args.save_mesh:
                camera_translation = cam_t.copy() - init_trans
                # print(init_trans, cam_t, joints)
                # exit()
                tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                tmesh.export(os.path.join(mesh_dir, f'{str(N).zfill(6)}_{is_right}.obj'))
        N += 1

    if args.render:
        if args.res_folder is not None:
            save_video(render_save_path, os.path.dirname(args.res_folder), args.out_folder)
            save_video(joint2d_save_path, os.path.dirname(args.res_folder), args.out_folder + '_2d')
            save_video(vit_save_path, os.path.dirname(args.res_folder), args.out_folder + '_vit')
        else:
            save_video(render_save_path, args.out_folder)
            save_video(joint2d_save_path, args.out_folder + '_2d')
            save_video(vit_save_path, args.out_folder + '_vit')

    # save_video(depth_save_path, 'dance_depth')
    for i in results_dict.keys():
        if len(results_dict[i]['tid']) > 1 and len(results_dict[i]['mano']) == 1:
            # print(results_dict[i]['tid'])
            assert (results_dict[i]['tid'] == [0,1]).all()
            if results_dict[i]['mano'][0]['is_right'] == 0:
                continue
            elif results_dict[i]['mano'][0]['is_right'] == 1:
                results_dict[i]['mano'].insert(0, -100)
                results_dict[i]['cam_trans'].insert(0, -100)
                assert np.array(results_dict[i]['extra_data']).shape == (1,21,3)
                d2 = results_dict[i]['extra_data'][0]
                results_dict[i]['extra_data'] = [-100, d2]
            # print(results_dict[i]['extra_data'])

    print('save to ', os.path.dirname(args.res_folder), args.res_folder)
    with open(args.res_folder, 'wb') as f:
        pickle.dump(results_dict, f)

if __name__ == '__main__':
    main()
