import ijson
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import PillowWriter
import sys

# ==========================================
# ===== GIF EXPORT CONFIGURATION =====
# ==========================================
OUTPUT_GIF_NAME = "shinjuku_paths_optimized.gif"

# -- Strict Web-Optimization Settings --
GIF_FPS = 10             # 10 FPS is plenty for a 10-second summary GIF
GIF_DPI = 60             # 60 DPI with 8x6 figsize = 480x360 resolution (Massive size reduction)
DATA_FRAME_SKIP = 2      # Process every 2nd frame (keeps physics stable)
CAPTURE_INTERVAL = 24    # 4750 total / 2 skip / 24 capture = ~99 total GIF frames

JSON_PATH = "data/SHINJUKU1/output/FULL_SHINJUKU1_2025-12-26_ID001.json"
SAT_IMAGE_PATH = "data/SHINJUKU1/sat_SHINJUKU1.png"
ALLOWED_CLASSES = {"car", "truck", "bus", "pedestrian"}

DT = 0.033
ENABLE_ALL_SMOOTHING = True      
ENABLE_POS_SMOOTHING = True      
ENABLE_HEADING_SMOOTHING = True  
GLOBAL_SMOOTHING = 0.1   
HEADING_SMOOTHING = 0.05 
L, MAX_STEER_DEG = 2.7, 35.0     

BG_ALPHA = 0.5           
LINE_WIDTH = 1.2         
LINE_ALPHA = 0.8         
MAX_HISTORY = 30         # CRITICAL: Keeps trails short. Long trails destroy GIF compression.

def safe_float(value, default=0.0):
    try: return float(value)
    except (TypeError, ValueError): return default

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Reduced figsize for smaller resolution
fig, ax = plt.subplots(figsize=(8, 6))
try:
    img = mpimg.imread(SAT_IMAGE_PATH)
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], alpha=BG_ALPHA)
except FileNotFoundError:
    print("No background found.")

ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Vehicle Paths (Kinematic Smoothing)", pad=10)
fig.tight_layout()

tracks = {} 
effective_dt = DT * DATA_FRAME_SKIP
writer = PillowWriter(fps=GIF_FPS)

print(f"Starting highly optimized rendering for {OUTPUT_GIF_NAME}...")

with writer.saving(fig, OUTPUT_GIF_NAME, dpi=GIF_DPI):
    with open(JSON_PATH, "rb") as f:
        frames = ijson.items(f, "frames.item")
        captured_frames = 0

        for frame_idx, frame in enumerate(frames):
            if frame_idx % DATA_FRAME_SKIP != 0:
                continue
                
            for obj in frame.get("objects", []):
                obj_class = obj.get("class", "").lower()
                if obj_class not in ALLOWED_CLASSES:
                    continue

                tid = obj["tracked_id"]
                z_x, z_y = safe_float(obj["sat_coords"][0]), safe_float(obj["sat_coords"][1])
                z_v = safe_float(obj.get("speed_kmh", 0)) / 3.6 
                z_theta = np.radians(safe_float(obj.get("heading", 0)))

                if tid not in tracks:
                    line, = ax.plot([], [], linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
                    tracks[tid] = {"line": line, "state": np.array([z_x, z_y, z_theta, z_v]), "x": [], "y": []}

                curr_s = tracks[tid]["state"]
                c_x, c_y, c_theta, c_v = curr_s

                if ENABLE_ALL_SMOOTHING:
                    theta_diff = normalize_angle(z_theta - c_theta)
                    steer_req = np.arctan((theta_diff / effective_dt) * (L / c_v)) if abs(c_v) > 0.1 else 0.0
                    steer = np.clip(steer_req, -np.radians(MAX_STEER_DEG), np.radians(MAX_STEER_DEG))

                    pred_theta = normalize_angle(c_theta + (c_v / L) * np.tan(steer) * effective_dt)
                    pred_x = c_x + c_v * np.cos(pred_theta) * effective_dt
                    pred_y = c_y + c_v * np.sin(pred_theta) * effective_dt

                    smooth_x = pred_x + GLOBAL_SMOOTHING * (z_x - pred_x) if ENABLE_POS_SMOOTHING else z_x
                    smooth_y = pred_y + GLOBAL_SMOOTHING * (z_y - pred_y) if ENABLE_POS_SMOOTHING else z_y
                    smooth_theta = pred_theta + HEADING_SMOOTHING * normalize_angle(z_theta - pred_theta) if ENABLE_HEADING_SMOOTHING else z_theta
                else:
                    smooth_x, smooth_y, smooth_theta = z_x, z_y, z_theta

                tracks[tid]["state"] = np.array([smooth_x, smooth_y, smooth_theta, z_v])
                tracks[tid]["x"].append(smooth_x)
                tracks[tid]["y"].append(smooth_y)

                if 0 < MAX_HISTORY < len(tracks[tid]["x"]):
                    tracks[tid]["x"].pop(0)
                    tracks[tid]["y"].pop(0)

                tracks[tid]["line"].set_data(tracks[tid]["x"], tracks[tid]["y"])

            # Capture frame
            if (frame_idx // DATA_FRAME_SKIP) % CAPTURE_INTERVAL == 0:
                writer.grab_frame()
                captured_frames += 1
                if captured_frames % 10 == 0:
                    sys.stdout.write(f"\rCaptured {captured_frames} GIF frames...")
                    sys.stdout.flush()

print(f"\nSuccess! Saved {OUTPUT_GIF_NAME}.")