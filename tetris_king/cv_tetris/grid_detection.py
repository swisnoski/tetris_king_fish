#!/usr/bin/env python3
"""
hybrid_tetris_detector.py
A script to detect and undistort a Tetris board, outputting only the final corner coordinates.
"""

import cv2
import numpy as np
import argparse
import os
from typing import Optional, List

# ---------- Config ----------
GRID_WIDTH = 10
GRID_HEIGHT = 20
BLOCK_PX = 50
WARPED_WIDTH = GRID_WIDTH * BLOCK_PX
WARPED_HEIGHT = GRID_HEIGHT * BLOCK_PX

# Hough parameters
HOUGH_P_RHO = 1
HOUGH_P_THETA = np.pi / 180
HOUGH_P_THRESHOLD = 90  
HOUGH_MIN_LINE_LENGTH = 200 
HOUGH_MAX_LINE_GAP = 10

# Manual corrections (necessary to correct initial detection errors)
BOUNDARY_SHRINK_PX = 6 
VERTICAL_OFFSET_PX = -5 
BOTTOM_EXPAND_PX = 10 
TOP_DROP_PX = 11 

# Top Horizontal Skew Correction: Increased to -10 for aggressive perspective correction.
TOP_SKEW_X = -10 

# ----------------------------

def try_import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except Exception:
        return None

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points in TL,TR,BR,BL order (float32)."""
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def adjust_rectangle_inward(rect: np.ndarray, shrink_px: int) -> np.ndarray:
    """Shrinks the ordered 4-point rectangle (TL, TR, BR, BL) inward by a fixed pixel amount."""
    if shrink_px == 0:
        return rect
    
    tl, tr, br, bl = rect
    
    # Calculate the center of the rectangle
    center_x = np.mean([tl[0], tr[0], br[0], bl[0]])
    center_y = np.mean([tl[1], tr[1], br[1], bl[1]])
    
    new_rect = np.zeros_like(rect)
    
    for i in range(4):
        dx = rect[i, 0] - center_x
        dy = rect[i, 1] - center_y
        
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 0:
            nx = dx / norm
            ny = dy / norm
            new_rect[i, 0] = rect[i, 0] - nx * shrink_px
            new_rect[i, 1] = rect[i, 1] - ny * shrink_px
        else:
            new_rect[i] = rect[i]
            
    return new_rect.astype(np.float32)


def bin_lines(data: np.ndarray, num_bins: int = 100) -> List[float]:
    """Helper to find the two extreme peaks in a line set (vertical or horizontal)."""
    if len(data) < 2:
        return []
    
    hist, bin_edges = np.histogram(data, bins=num_bins)
    smoothed_hist = np.convolve(hist, np.ones(3)/3, mode='same')
    
    peak_threshold = np.max(smoothed_hist) * 0.20 
    peak_indices = np.where(smoothed_hist > peak_threshold)[0]
    
    if len(peak_indices) < 2:
        return [np.min(data), np.max(data)]

    low_peak_idx = peak_indices[0]
    for idx in peak_indices:
        if bin_edges[idx+1] - bin_edges[low_peak_idx] > (bin_edges[-1] - bin_edges[0]) / 15: 
            low_peak_idx = idx
            break
            
    low_bin_start = bin_edges[low_peak_idx]
    low_bin_end = bin_edges[low_peak_idx + 1]
    low_cluster_data = data[(data >= low_bin_start) & (data < low_bin_end)]
    low_val = np.mean(low_cluster_data) if low_cluster_data.size > 0 else np.min(data)
    
    high_peak_idx = peak_indices[-1]
    high_bin_start = bin_edges[high_peak_idx]
    high_bin_end = bin_edges[high_peak_idx + 1]
    high_cluster_data = data[(data >= high_bin_start) & (data < high_bin_end)]
    high_val = np.mean(high_cluster_data) if high_cluster_data.size > 0 else np.max(data)
    
    return [low_val, high_val]


def detect_board_hough(img: np.ndarray) -> Optional[np.ndarray]:
    """Attempt to find the board rectangle using Hough lines."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 80, 180)

    lines = cv2.HoughLinesP(edges,
                            rho=HOUGH_P_RHO,
                            theta=HOUGH_P_THETA,
                            threshold=HOUGH_P_THRESHOLD,
                            minLineLength=HOUGH_MIN_LINE_LENGTH,
                            maxLineGap=HOUGH_MAX_LINE_GAP)

    if lines is None:
        return None

    vertical_x = []
    horizontal_y = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) < abs(dy) * 0.3 and abs(dy) > h * 0.5:  
            vertical_x.append((x1+x2)/2)
        elif abs(dy) < abs(dx) * 0.5: 
            horizontal_y.append((y1+y2)/2)

    if len(vertical_x) < 2 or len(horizontal_y) < 2:
        return None

    vx = np.array(vertical_x)
    hx = np.array(horizontal_y)
    
    vx_edges = bin_lines(vx)
    hx_edges = bin_lines(hx)
    
    if not vx_edges or not hx_edges:
        return None
        
    left, right = min(vx_edges), max(vx_edges)
    top, bottom = min(hx_edges), max(hx_edges)
    
    # Clamp to image
    left = max(0, min(w-1, left))
    right = max(0, min(w-1, right))
    top = max(0, min(h-1, top))
    bottom = max(0, min(h-1, bottom))

    # Make a rectangle (order TL, TR, BR, BL)
    tl = [left, top]
    tr = [right, top]
    br = [right, bottom]
    bl = [left, bottom]

    rect = np.array([tl, tr, br, bl], dtype=np.float32)
    return rect

def detect_board_colormask(img: np.ndarray) -> Optional[np.ndarray]:
    """Fallback: detect board by color cluster / mask and find largest rectangular contour."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    ranges = [
        ((0, 0, 0), (180, 255, 60)), 
    ]
    
    best_cnt = None
    best_area = 0
    for low, high in ranges:
        low = np.array(low, dtype=np.uint8)
        high = np.array(high, dtype=np.uint8)
        m = cv2.inRange(hsv, low, high)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > best_area:
            best_area = area
            best_cnt = c
    if best_cnt is None or best_area < 1000:
        return None

    rect = cv2.minAreaRect(best_cnt)
    w_min_area, h_min_area = rect[1]
    
    if w_min_area > h_min_area:
        w_min_area, h_min_area = h_min_area, w_min_area
        
    aspect_ratio = w_min_area / h_min_area
    
    if aspect_ratio < 0.3 or aspect_ratio > 0.7:
        return None
    
    box = cv2.boxPoints(rect)
    box = order_points(np.array(box))
    return box

def try_yolo_board_detect(img_path: str) -> Optional[np.ndarray]:
    YOLO = try_import_ultralytics()
    if YOLO is None:
        return None
    
    model_path = os.environ.get("YOLO_TETRIS_BOARD_MODEL", None)
    if model_path is None:
        return None
    try:
        model = YOLO(model_path)
        results = model.predict(source=img_path, save=False, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
            
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        idx = int(np.argmax(scores))
        x1,y1,x2,y2 = boxes[idx]
        rect = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        return rect
    except Exception:
        return None

def normalize_vertical_alignment(rect: np.ndarray) -> np.ndarray:
    """
    Forces the top edge (TL, TR) and bottom edge (BL, BR) to be perfectly horizontal.
    """
    y_top_avg = (rect[0, 1] + rect[1, 1]) / 2.0
    y_bottom_avg = (rect[2, 1] + rect[3, 1]) / 2.0
    
    rect[0, 1] = y_top_avg  # TL
    rect[1, 1] = y_top_avg  # TR
    rect[2, 1] = y_bottom_avg # BR
    rect[3, 1] = y_bottom_avg # BL
    
    return rect

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", "-i", required=False, default="/home/satchel/tetris_king_fish/tetris_king/cv_tetris/assets/start_tetris_cleaned.jpg",
                   help="Path to image")
    p.add_argument("--use-yolo", action="store_true", help="Try YOLO board detection (optional)")
    args = p.parse_args()

    # Load image (or create placeholder if path is bad/missing)
    try:
        img = cv2.imread(args.image)
        if img is None:
             img = np.zeros((1080, 1920, 3), dtype=np.uint8) 
    except Exception:
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # 1. Detection
    board_rect = detect_board_hough(img)

    if board_rect is not None:
        board_rect = order_points(board_rect)

    if board_rect is None and args.use_yolo:
        board_rect = try_yolo_board_detect(args.image) 
        if board_rect is not None:
            board_rect = order_points(board_rect)

    if board_rect is None:
        board_rect = detect_board_colormask(img) 

    if board_rect is None:
        print("Failed to find board by any method.")
        return

    # 2. Automated and Manual Corrections

    # Change 1: Shrink boundary inward
    if BOUNDARY_SHRINK_PX > 0:
        board_rect = adjust_rectangle_inward(board_rect, BOUNDARY_SHRINK_PX)
    
    # Change 2: Apply vertical offset
    if VERTICAL_OFFSET_PX != 0:
        board_rect[:, 1] += VERTICAL_OFFSET_PX  
    
    # Change 3: Expand bottom edge
    if BOTTOM_EXPAND_PX != 0:
        board_rect[2, 1] += BOTTOM_EXPAND_PX  
        board_rect[3, 1] += BOTTOM_EXPAND_PX  

    # Change 4: Specific top drop
    if TOP_DROP_PX != 0:
        board_rect[0, 1] += TOP_DROP_PX  
        board_rect[1, 1] += TOP_DROP_PX  
    
    # Change 5: Top horizontal skew correction (Undistortion fix)
    if TOP_SKEW_X != 0:
        board_rect[1, 0] += TOP_SKEW_X 
        
    # Change 7: Automatic vertical undistortion (Leveling fix)
    board_rect = normalize_vertical_alignment(board_rect)

    # 3. Output Final Coordinates

    # Convert coordinates to integers for clean output
    final_coords = board_rect.astype(int)

    # Output only the four corner coordinates in the requested format
    for i, label in enumerate(["TL", "TR", "BR", "BL"]):
        print(f"{final_coords[i][0]},{final_coords[i][1]}")


if __name__ == "__main__":
    main()