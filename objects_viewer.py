import ijson
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import io
import os
import sys

# ==========================================
# ===== CONFIGURATION & TUNING VARIABLES =====
# ==========================================

# -- File Paths --
BASE_DIR = "data/SHINJUKU1"
PROJ_JSON_PATH = os.path.join(BASE_DIR, "G_projection_SHINJUKU1.json")
DATA_PATH = os.path.join(BASE_DIR, "output/FULL_SHINJUKU1_2025-12-26_ID001.json")

# -- Data & Timing --
DT = 0.033               # Base time step of the data (e.g., 30 FPS = 0.033s)
DATA_FRAME_SKIP = 1      # 1 = process every frame, 2 = process every 2nd frame, etc.

# -- Filter & Smoothing Toggles --
ENABLE_ALL_SMOOTHING = True      # MASTER TOGGLE. Bypasses physics if False.
ENABLE_POS_SMOOTHING = True      # Snap x/y coords instantly to raw data if False.
ENABLE_HEADING_SMOOTHING = True  # Snap vehicle rotation instantly to raw data if False.

GLOBAL_SMOOTHING = 0.1   # Base Position blend: 1.0 = raw data, 0.01 = heavy smoothing
HEADING_SMOOTHING = 0.05 # Heading blend: How quickly smoothed heading snaps to raw
JUMP_GATE_METERS = 2.0   # If raw data jumps more than this physically, reject the noise

# -- Kinematic Bicycle Physics --
L = 2.7                  # Wheelbase in meters
MAX_STEER_DEG = 35.0     # Maximum physical steering angle limits

# -- Visual / UI Controls --
DRAW_INTERVAL = 5        # Update the plot every N processed frames
LIFESPAN = 50            # Keep only the most recent N frames of each path
MAX_SILENCE = 10         # Frames to wait before starting graceful decay
DECAY_RATE = 1           # Points removed from the path per inactive frame
HEAD_MARKER_SIZE = 4     # Size of the latest head dot
COLOR_BY_CLASS = True    # Set False to color tracks by id instead

# ALLOWED_CLASSES = {"car", "truck", "bus"}
ALLOWED_CLASSES = {"car", "truck", "bus", "pedestrian"}
CLASS_COLOR_MAP = {
    "car": "cyan",
    "truck": "orange",
    "bus": "lime",
    "pedestrian": "magenta"
}
ID_PALETTE = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "gold",
    "magenta", "lime", "dodgerblue", "tomato", "orchid",
    "seagreen", "navy", "darkorange", "royalblue", "crimson"
]

# ==========================================
# ===== DYNAMIC CONFIG & HELPERS =====
# ==========================================

with open(PROJ_JSON_PATH, 'r') as f:
    proj_data = json.load(f)

SAT_PATH = os.path.join(BASE_DIR, proj_data["inputs"]["sat_path"])
SVG_PATH = os.path.join(BASE_DIR, proj_data["inputs"]["layout_path"])
PX_PER_METER = proj_data["parallax"]["px_per_meter"]
CAM_SAT_POS = np.array([
    proj_data["parallax"]["x_cam_coords_sat"], 
    proj_data["parallax"]["y_cam_coords_sat"]
])

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_track_color(tid, obj_class):
    if COLOR_BY_CLASS:
        return CLASS_COLOR_MAP.get(obj_class, "tab:blue")
    return ID_PALETTE[hash(tid) % len(ID_PALETTE)]

def load_svg_as_image(path, target_width, target_height):
    drawing = svg2rlg(path)
    orig_w, orig_h = drawing.width, drawing.height
    scale_x = target_width / orig_w
    scale_y = target_height / orig_h
    drawing.width, drawing.height = target_width, target_height
    drawing.scale(scale_x, scale_y)
    img_data = renderPM.drawToString(drawing, fmt="PNG")
    return mpimg.imread(io.BytesIO(img_data))

# ==========================================
# ===== SETUP FIGURE =====
# ==========================================

plt.ion()
fig, ax = plt.subplots(figsize=(12, 10))
fig_num = fig.number
plt.subplots_adjust(bottom=0.2)

sat_img = mpimg.imread(SAT_PATH)
h, w = sat_img.shape[:2]

print(f"Rendering SVG Map: {SVG_PATH}...")
try:
    svg_img = load_svg_as_image(SVG_PATH, w, h)
except Exception as e:
    print(f"Failed to load SVG: {e}. Plotting without it.")
    svg_img = np.zeros((h, w, 4)) # Blank fallback

# Display Layers
sat_layer = ax.imshow(sat_img, extent=[0, w, h, 0], alpha=0.0, zorder=1)
svg_layer = ax.imshow(svg_img, extent=[0, w, h, 0], alpha=1.0, zorder=2)

# ===== SLIDERS =====
ax_sat = plt.axes([0.2, 0.05, 0.6, 0.03])
ax_svg = plt.axes([0.2, 0.10, 0.6, 0.03])
s_sat = Slider(ax_sat, 'Sat Alpha', 0.0, 1.0, valinit=0.0)
s_svg = Slider(ax_svg, 'SVG Alpha', 0.0, 1.0, valinit=1.0)

def update_sliders(val):
    sat_layer.set_alpha(s_sat.val)
    svg_layer.set_alpha(s_svg.val)
    fig.canvas.draw_idle()

s_sat.on_changed(update_sliders)
s_svg.on_changed(update_sliders)

# ==========================================
# ===== MAIN LOOP =====
# ==========================================

tracks = {}
effective_dt = DT * DATA_FRAME_SKIP

print("Starting simulation... (Close the plot window or press Ctrl+C to exit)")

try:
    with open(DATA_PATH, "rb") as f:
        frames = ijson.items(f, "frames.item")
        
        for frame_idx, frame in enumerate(frames):
            
            # --- TERMINATION CHECK ---
            if not plt.fignum_exists(fig_num):
                print("\nPlot window closed. Terminating script.")
                break
                
            # --- FRAME SKIPPING LOGIC ---
            if frame_idx % DATA_FRAME_SKIP != 0:
                continue

            seen_tids = set()
            for obj in frame.get("objects", []):
                obj_class = obj.get("class", "").lower()
                if obj_class not in ALLOWED_CLASSES:
                    continue

                tid = obj["tracked_id"]
                seen_tids.add(tid)
                
                # Raw Measurements
                z_x = safe_float(obj["sat_coords"][0])
                z_y = safe_float(obj["sat_coords"][1])
                z_v = safe_float(obj.get("speed_kmh", 0)) / 3.6 
                z_theta = np.radians(safe_float(obj.get("heading", 0)))
                z_pos = np.array([z_x, z_y])

                if tid not in tracks:
                    color = get_track_color(tid, obj_class)
                    line, = ax.plot([], [], linewidth=1.5, alpha=0.8, zorder=3, color=color)
                    head, = ax.plot([], [], marker='o', markersize=HEAD_MARKER_SIZE, color=color, zorder=4)
                    tracks[tid] = {
                        "line": line, "head": head,
                        "state": np.array([z_x, z_y, z_theta, z_v]), # Kinematic state
                        "x": [z_x], "y": [z_y],
                        "inactive_frames": 0, "class": obj_class
                    }
                    tracks[tid]["line"].set_data(tracks[tid]["x"], tracks[tid]["y"])
                    tracks[tid]["head"].set_data([z_x], [z_y])
                    continue

                # --- KINEMATIC BICYCLE UPDATE ---
                curr_s = tracks[tid]["state"]
                c_x, c_y, c_theta, c_v = curr_s

                # Dynamic Alpha based on Distance and Jump-Gate
                dist_to_cam = np.linalg.norm(z_pos - CAM_SAT_POS)
                dynamic_gain = GLOBAL_SMOOTHING * np.clip(500/max(dist_to_cam, 1), 0.2, 1.0)
                
                # Physical jump gate: if it jumps wildly, reject the noise heavily
                if np.linalg.norm(z_pos - np.array([c_x, c_y])) / PX_PER_METER > JUMP_GATE_METERS:
                    dynamic_gain *= 0.1

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
                        smooth_x = pred_x + dynamic_gain * (z_x - pred_x)
                        smooth_y = pred_y + dynamic_gain * (z_y - pred_y)
                    else:
                        smooth_x, smooth_y = z_x, z_y

                    if ENABLE_HEADING_SMOOTHING:
                        smooth_theta = pred_theta + HEADING_SMOOTHING * normalize_angle(z_theta - pred_theta)
                    else:
                        smooth_theta = z_theta
                
                else:
                    smooth_x, smooth_y, smooth_theta = z_x, z_y, z_theta

                # Save Data
                tracks[tid]["state"] = np.array([smooth_x, smooth_y, smooth_theta, z_v])
                tracks[tid]["inactive_frames"] = 0
                tracks[tid]["x"].append(smooth_x)
                tracks[tid]["y"].append(smooth_y)
                
                # Truncate to Lifespan
                tracks[tid]["x"] = tracks[tid]["x"][-LIFESPAN:]
                tracks[tid]["y"] = tracks[tid]["y"][-LIFESPAN:]
                
                tracks[tid]["line"].set_data(tracks[tid]["x"], tracks[tid]["y"])
                tracks[tid]["head"].set_data([smooth_x], [smooth_y])

            # --- TRACK DECAY (FADING) ---
            for tid, data in list(tracks.items()):
                if tid in seen_tids:
                    continue
                data["inactive_frames"] += 1
                if data["inactive_frames"] <= MAX_SILENCE:
                    continue

                if len(data["x"]) > 0:
                    data["x"] = data["x"][DECAY_RATE:]
                if len(data["y"]) > 0:
                    data["y"] = data["y"][DECAY_RATE:]

                remaining_ratio = len(data["x"]) / max(1, LIFESPAN)
                alpha = max(0.1, 0.8 * remaining_ratio)
                
                if data["x"] and data["y"]:
                    data["line"].set_alpha(alpha)
                    data["line"].set_data(data["x"], data["y"])
                    data["head"].set_data([data["x"][-1]], [data["y"][-1]])
                    data["head"].set_alpha(alpha)
                else:
                    # Remove completely dead tracks
                    data["line"].set_data([], [])
                    data["head"].set_data([], [])
                    del tracks[tid]

            # --- RENDER ---
            if (frame_idx // DATA_FRAME_SKIP) % DRAW_INTERVAL == 0:
                ax.set_title(f"Shinjuku 1 | Kinematic Smoothing | Frame: {frame_idx}")
                plt.draw()
                plt.pause(0.001)

    # --- FINISH ---
    print("\nAnimation finished! The plot will remain open until you close it.")
    plt.ioff()
    plt.show()

except KeyboardInterrupt:
    print("\nProcess interrupted by user (Ctrl+C). Exiting.")
    plt.close('all')
    sys.exit(0)

except FileNotFoundError as e:
    print(f"\nError loading a required file: {e}")
    sys.exit(1)