import cv2
import numpy as np
from skimage.feature import blob_log
import matplotlib.pyplot as plt
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    denoised = cv2.fastNlMeansDenoising(img, None, h=12, templateWindowSize=7, searchWindowSize=20)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(9, 9))
    enhanced = clahe.apply(denoised)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -3
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return img, denoised, enhanced, thresh

def detect_stars(enhanced):
    blobs = blob_log(enhanced, max_sigma=3, num_sigma=90, threshold=0.05)
    stars = [(b[1], b[0], b[2]) for b in blobs]
    return stars, len(stars)

def line_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def merge_streaks(streaks, angle_thresh=5, dist_thresh=10):
    """Merge collinear and nearby streaks"""
    merged = []
    used = [False] * len(streaks)
    for i in range(len(streaks)):
        if used[i]:
            continue
        _, _, x1, y1, x2, y2 = streaks[i]
        angle1 = line_angle(x1, y1, x2, y2)
        pts = [(x1, y1), (x2, y2)]
        for j in range(i + 1, len(streaks)):
            if used[j]:
                continue
            _, _, x3, y3, x4, y4 = streaks[j]
            angle2 = line_angle(x3, y3, x4, y4)
            if abs(angle1 - angle2) > angle_thresh:
                continue
            dists = [
                np.hypot(x1 - x3, y1 - y3),
                np.hypot(x1 - x4, y1 - y4),
                np.hypot(x2 - x3, y2 - y3),
                np.hypot(x2 - x4, y2 - y4)
            ]
            if min(dists) < dist_thresh:
                pts.extend([(x3, y3), (x4, y4)])
                used[j] = True
        pts = np.array(pts, dtype=np.float32)
        mean = pts.mean(axis=0)
        pts_centered = pts - mean
        _, _, vt = np.linalg.svd(pts_centered)
        direction = vt[0]
        projections = pts_centered @ direction
        p_min = mean + direction * projections.min()
        p_max = mean + direction * projections.max()
        cx = (p_min[0] + p_max[0]) / 2
        cy = (p_min[1] + p_max[1]) / 2
        merged.append((
            cx, cy,
            int(p_min[0]), int(p_min[1]),
            int(p_max[0]), int(p_max[1])
        ))
        used[i] = True
    return merged

def detect_streaks(thresh):
    lines = cv2.HoughLinesP(
        thresh, rho=1, theta=np.pi/180, threshold=30,
        minLineLength=22, maxLineGap=2
    )
    streaks = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            streaks.append((cx, cy, x1, y1, x2, y2))
    streaks = merge_streaks(streaks)
    return streaks, len(streaks) if lines is not None else 0

def visualize_results(img, denoised, enhanced, thresh, stars, streaks, filename, output_folder):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Detection Results: {filename}", fontsize=14, fontweight='bold')  
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(denoised, cmap='gray')
    axes[0, 1].set_title("Denoised Image")
    axes[0, 1].axis('off')
    axes[0, 2].imshow(enhanced, cmap='gray')
    axes[0, 2].set_title("Enhanced (CLAHE)")
    axes[0, 2].axis('off')
    axes[1, 0].imshow(thresh, cmap='gray')
    axes[1, 0].set_title("Thresholded Image")
    axes[1, 0].axis('off')
    star_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, sigma in stars:
        cv2.circle(star_img, (int(x), int(y)), int(sigma*1.5), (0, 255, 0), 2)
    axes[1, 1].imshow(cv2.cvtColor(star_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Stars Detected ({len(stars)})")
    axes[1, 1].axis('off')
    streak_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cx, cy, x1, y1, x2, y2 in streaks:
        cv2.line(streak_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.circle(streak_img, (int(cx), int(cy)), 3, (0, 255, 255), -1)
    axes[1, 2].imshow(cv2.cvtColor(streak_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"Streaks Detected ({len(streaks)})")
    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_detected.jpg"))
    plt.close(fig)

def process_datasets(dataset_folder="Datasets_Assessment"):
    results = []
    for file in os.listdir(dataset_folder):
        img_path = os.path.join(dataset_folder, file)
        if not os.path.isfile(img_path) or not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        print(f"\nProcessing: {file}")
        try:
            img, denoised, enhanced, thresh = preprocess_image(img_path)
            stars, star_count = detect_stars(enhanced)
            streaks, streak_count = detect_streaks(thresh)
            visualize_results(img, denoised, enhanced, thresh, stars, streaks, file, OUTPUT_DIR)
            for x, y, _ in stars:
                results.append(["star", file, x, y])
            for x, y, _, _, _, _ in streaks:
                results.append(["object", file, x, y])
            print(f"Stars detected: {star_count}, Objects detected: {streak_count}")
            df = pd.DataFrame(results, columns=["feature", "image", "centroid_x", "centroid_y"])
            csv_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_detected.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nAll detections are saved to: {csv_path}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    process_datasets()
