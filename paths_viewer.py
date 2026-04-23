import ijson
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

# ==========================================
# ===== CONFIGURATION & TUNING VARIABLES =====
# ==========================================

# -- File Paths --
JSON_PATH = "data/SHINJUKU1/output/FULL_SHINJUKU1_2025-12-26_ID001.json"
SAT_IMAGE_PATH = "data/SHINJUKU1/sat_SHINJUKU1.png"
# ALLOWED_CLASSES = {"car", "truck", "bus"}
ALLOWED_CLASSES = {"car", "truck", "bus", "pedestrian"}

# -- Data & Timing --
DT = 0.033               # Base time step of the data (e.g., 30 FPS = 0.033s)
DATA_FRAME_SKIP = 1      # 1 = process every frame, 2 = process every 2nd frame, etc.

# -- Filter & Smoothing Toggles --
ENABLE_ALL_SMOOTHING = True      # MASTER TOGGLE. Bypasses physics if False.
ENABLE_POS_SMOOTHING = True      # Snap x/y coords instantly to raw data if False.
ENABLE_HEADING_SMOOTHING = True  # Snap vehicle rotation instantly to raw data if False.

GLOBAL_SMOOTHING = 0.1   # Position blend: 1.0 = raw data, 0.01 = heavy smoothing
HEADING_SMOOTHING = 0.05 # Heading blend: How quickly smoothed heading snaps to raw

# -- Kinematic Bicycle Physics --
L = 2.7                  # Wheelbase in meters
MAX_STEER_DEG = 35.0     # Maximum physical steering angle limits

# -- Visual / UI Controls --
BG_ALPHA = 0.6           # Opacity of the background satellite image
LINE_WIDTH = 1.2         # Thickness of the tracking paths
LINE_ALPHA = 0.9         # Opacity of the tracking paths
DRAW_INTERVAL = 50       # Update the plot every N processed frames
MAX_HISTORY = 0          # Points to keep. 0 = keep the full path forever.

# ==========================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def normalize_angle(angle):
    """Keeps an angle safely within -pi and pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ===== SETUP PLOT =====
plt.ion()
fig, ax = plt.subplots(figsize=(12, 10))
fig_num = fig.number # Store figure number to check if it gets closed

try:
    img = mpimg.imread(SAT_IMAGE_PATH)
    img_height, img_width = img.shape[:2]
    ax.imshow(img, extent=[0, img_width, img_height, 0], alpha=BG_ALPHA)
except FileNotFoundError:
    print(f"Warning: Could not find image at {SAT_IMAGE_PATH}. Plotting without background.")

tracks = {} 
effective_dt = DT * DATA_FRAME_SKIP

print("Starting simulation... (Close the plot window or press Ctrl+C to exit)")

try:
    with open(JSON_PATH, "rb") as f:
        frames = ijson.items(f, "frames.item")

        for frame_idx, frame in enumerate(frames):
            
            # --- TERMINATION CHECK ---
            # If the user closes the plot window, break the loop and exit
            if not plt.fignum_exists(fig_num):
                print("\nPlot window closed. Terminating script.")
                break
            
            # --- FRAME SKIPPING LOGIC ---
            if frame_idx % DATA_FRAME_SKIP != 0:
                continue
                
            for obj in frame.get("objects", []):
                obj_class = obj.get("class", "").lower()
                if obj_class not in ALLOWED_CLASSES:
                    continue

                tid = obj["tracked_id"]
                z_x_raw, z_y_raw = obj["sat_coords"] 
                z_x = safe_float(z_x_raw)
                z_y = safe_float(z_y_raw)
                z_v = safe_float(obj.get("speed_kmh", 0)) / 3.6 
                z_theta = np.radians(safe_float(obj.get("heading", 0)))

                if tid not in tracks:
                    line, = ax.plot([], [], linewidth=LINE_WIDTH, alpha=LINE_ALPHA, label=f"ID {tid}")
                    tracks[tid] = {
                        "line": line,
                        "state": np.array([z_x, z_y, z_theta, z_v]),
                        "x": [], "y": []
                    }

                # --- KINEMATIC BICYCLE UPDATE ---
                curr_s = tracks[tid]["state"]
                c_x, c_y, c_theta, c_v = curr_s

                if ENABLE_ALL_SMOOTHING:
                    theta_diff = normalize_angle(z_theta - c_theta)
                    
                    if abs(c_v) > 0.1:
                        steer_req = np.arctan((theta_diff / effective_dt) * (L / c_v))
                    else:
                        steer_req = 0.0

                    max_steer_rad = np.radians(MAX_STEER_DEG)
                    steer = np.clip(steer_req, -max_steer_rad, max_steer_rad)

                    pred_theta = c_theta + (c_v / L) * np.tan(steer) * effective_dt
                    pred_theta = normalize_angle(pred_theta)
                    
                    pred_x = c_x + c_v * np.cos(pred_theta) * effective_dt
                    pred_y = c_y + c_v * np.sin(pred_theta) * effective_dt

                    if ENABLE_POS_SMOOTHING:
                        smooth_x = pred_x + GLOBAL_SMOOTHING * (z_x - pred_x)
                        smooth_y = pred_y + GLOBAL_SMOOTHING * (z_y - pred_y)
                    else:
                        smooth_x, smooth_y = z_x, z_y

                    if ENABLE_HEADING_SMOOTHING:
                        smooth_theta = pred_theta + HEADING_SMOOTHING * normalize_angle(z_theta - pred_theta)
                    else:
                        smooth_theta = z_theta
                
                else:
                    smooth_x, smooth_y, smooth_theta = z_x, z_y, z_theta

                tracks[tid]["state"] = np.array([smooth_x, smooth_y, smooth_theta, z_v])
                tracks[tid]["x"].append(smooth_x)
                tracks[tid]["y"].append(smooth_y)

                if MAX_HISTORY > 0 and len(tracks[tid]["x"]) > MAX_HISTORY:
                    tracks[tid]["x"].pop(0)
                    tracks[tid]["y"].pop(0)

                tracks[tid]["line"].set_data(tracks[tid]["x"], tracks[tid]["y"])

            # Draw Interval
            if (frame_idx // DATA_FRAME_SKIP) % DRAW_INTERVAL == 0:
                ax.set_title(f"Shinjuku 1 - Kinematic Smoothing - Processed Frame: {frame_idx}")
                plt.draw()
                plt.pause(0.001)

        print("\nAnimation finished! The plot will remain open until you close it.")
        plt.ioff()      # Turn off interactive mode
        plt.show()      # Block execution and keep the window open

except KeyboardInterrupt:
    print("\nProcess interrupted by user (Ctrl+C). Exiting.")
    plt.close('all')
    sys.exit(0)

except FileNotFoundError:
    print(f"\nError: Could not find the JSON file at {JSON_PATH}")
    sys.exit(1)