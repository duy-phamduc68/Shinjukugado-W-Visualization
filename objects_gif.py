import ijson
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import PillowWriter
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import io
import os
import sys

# ==========================================
# ===== GIF EXPORT CONFIGURATION =====
# ==========================================
OUTPUT_GIF_NAME = "shinjuku_objects_optimized.gif"

# -- Strict Web-Optimization Settings --
GIF_FPS = 10             
GIF_DPI = 60             
DATA_FRAME_SKIP = 2      
CAPTURE_INTERVAL = 24    

BASE_DIR = "data/SHINJUKU1"
PROJ_JSON_PATH = os.path.join(BASE_DIR, "G_projection_SHINJUKU1.json")
DATA_PATH = os.path.join(BASE_DIR, "output/FULL_SHINJUKU1_2025-12-26_ID001.json")

DT = 0.033 
ENABLE_ALL_SMOOTHING = True      
ENABLE_POS_SMOOTHING = True      
ENABLE_HEADING_SMOOTHING = True  
GLOBAL_SMOOTHING = 0.1   
HEADING_SMOOTHING = 0.05 
JUMP_GATE_METERS = 2.0   
L, MAX_STEER_DEG = 2.7, 35.0     

LIFESPAN = 30            # Dropped from 50 to 30 to help with GIF compression
MAX_SILENCE = 5          
DECAY_RATE = 2           
HEAD_MARKER_SIZE = 3     

# Visual Overrides for GIF
STATIC_SAT_ALPHA = 0.15
STATIC_SVG_ALPHA = 1.0

ALLOWED_CLASSES = {"car", "truck", "bus", "pedestrian"}
CLASS_COLOR_MAP = {"car": "cyan", "truck": "orange", "bus": "lime", "pedestrian": "magenta"}

with open(PROJ_JSON_PATH, 'r') as f:
    proj_data = json.load(f)

SAT_PATH = os.path.join(BASE_DIR, proj_data["inputs"]["sat_path"])
SVG_PATH = os.path.join(BASE_DIR, proj_data["inputs"]["layout_path"])
PX_PER_METER = proj_data["parallax"]["px_per_meter"]
CAM_SAT_POS = np.array([proj_data["parallax"]["x_cam_coords_sat"], proj_data["parallax"]["y_cam_coords_sat"]])

def safe_float(v):
    try: return float(v)
    except: return 0.0

def normalize_angle(a): return (a + np.pi) % (2 * np.pi) - np.pi

def load_svg_as_image(path, tw, th):
    drawing = svg2rlg(path)
    drawing.scale(tw / drawing.width, th / drawing.height)
    drawing.width, drawing.height = tw, th
    return mpimg.imread(io.BytesIO(renderPM.drawToString(drawing, fmt="PNG")))

# Reduced figsize for smaller resolution
fig, ax = plt.subplots(figsize=(8, 6))
sat_img = mpimg.imread(SAT_PATH)
h, w = sat_img.shape[:2]

try: svg_img = load_svg_as_image(SVG_PATH, w, h)
except: svg_img = np.zeros((h, w, 4)) 

ax.imshow(sat_img, extent=[0, w, h, 0], alpha=STATIC_SAT_ALPHA, zorder=1)
ax.imshow(svg_img, extent=[0, w, h, 0], alpha=STATIC_SVG_ALPHA, zorder=2)

ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Active Objects (SVG Map + Decay)", pad=10)
fig.tight_layout()

tracks = {}
effective_dt = DT * DATA_FRAME_SKIP
writer = PillowWriter(fps=GIF_FPS)

print(f"Starting highly optimized rendering for {OUTPUT_GIF_NAME}...")

with writer.saving(fig, OUTPUT_GIF_NAME, dpi=GIF_DPI):
    with open(DATA_PATH, "rb") as f:
        frames = ijson.items(f, "frames.item")
        captured_frames = 0
        
        for frame_idx, frame in enumerate(frames):
            if frame_idx % DATA_FRAME_SKIP != 0:
                continue

            seen_tids = set()
            for obj in frame.get("objects", []):
                obj_class = obj.get("class", "").lower()
                if obj_class not in ALLOWED_CLASSES: continue

                tid = obj["tracked_id"]
                seen_tids.add(tid)
                z_x, z_y = safe_float(obj["sat_coords"][0]), safe_float(obj["sat_coords"][1])
                z_v = safe_float(obj.get("speed_kmh", 0)) / 3.6 
                z_theta = np.radians(safe_float(obj.get("heading", 0)))
                z_pos = np.array([z_x, z_y])

                if tid not in tracks:
                    color = CLASS_COLOR_MAP.get(obj_class, "tab:blue")
                    line, = ax.plot([], [], linewidth=1.5, alpha=0.8, zorder=3, color=color)
                    head, = ax.plot([], [], marker='o', markersize=HEAD_MARKER_SIZE, color=color, zorder=4)
                    tracks[tid] = {
                        "line": line, "head": head, "state": np.array([z_x, z_y, z_theta, z_v]), 
                        "x": [z_x], "y": [z_y], "inactive_frames": 0
                    }
                    continue

                curr_s = tracks[tid]["state"]
                c_x, c_y, c_theta, c_v = curr_s

                dist_to_cam = np.linalg.norm(z_pos - CAM_SAT_POS)
                dynamic_gain = GLOBAL_SMOOTHING * np.clip(500/max(dist_to_cam, 1), 0.2, 1.0)
                if np.linalg.norm(z_pos - np.array([c_x, c_y])) / PX_PER_METER > JUMP_GATE_METERS:
                    dynamic_gain *= 0.1

                if ENABLE_ALL_SMOOTHING:
                    theta_diff = normalize_angle(z_theta - c_theta)
                    steer_req = np.arctan((theta_diff / effective_dt) * (L / c_v)) if abs(c_v) > 0.1 else 0.0
                    steer = np.clip(steer_req, -np.radians(MAX_STEER_DEG), np.radians(MAX_STEER_DEG))

                    pred_theta = normalize_angle(c_theta + (c_v / L) * np.tan(steer) * effective_dt)
                    pred_x, pred_y = c_x + c_v * np.cos(pred_theta) * effective_dt, c_y + c_v * np.sin(pred_theta) * effective_dt

                    smooth_x = pred_x + dynamic_gain * (z_x - pred_x) if ENABLE_POS_SMOOTHING else z_x
                    smooth_y = pred_y + dynamic_gain * (z_y - pred_y) if ENABLE_POS_SMOOTHING else z_y
                    smooth_theta = pred_theta + HEADING_SMOOTHING * normalize_angle(z_theta - pred_theta) if ENABLE_HEADING_SMOOTHING else z_theta
                else:
                    smooth_x, smooth_y, smooth_theta = z_x, z_y, z_theta

                tracks[tid]["state"] = np.array([smooth_x, smooth_y, smooth_theta, z_v])
                tracks[tid]["inactive_frames"] = 0
                tracks[tid]["x"].append(smooth_x)
                tracks[tid]["y"].append(smooth_y)
                
                tracks[tid]["x"] = tracks[tid]["x"][-LIFESPAN:]
                tracks[tid]["y"] = tracks[tid]["y"][-LIFESPAN:]
                
                tracks[tid]["line"].set_data(tracks[tid]["x"], tracks[tid]["y"])
                tracks[tid]["head"].set_data([smooth_x], [smooth_y])

            # Track Decay
            for tid, data in list(tracks.items()):
                if tid in seen_tids: continue
                data["inactive_frames"] += 1
                if data["inactive_frames"] <= MAX_SILENCE: continue

                if data["x"]: data["x"] = data["x"][DECAY_RATE:]
                if data["y"]: data["y"] = data["y"][DECAY_RATE:]

                alpha = max(0.1, 0.8 * (len(data["x"]) / max(1, LIFESPAN)))
                
                if data["x"] and data["y"]:
                    data["line"].set_alpha(alpha)
                    data["line"].set_data(data["x"], data["y"])
                    data["head"].set_data([data["x"][-1]], [data["y"][-1]])
                    data["head"].set_alpha(alpha)
                else:
                    data["line"].set_data([], [])
                    data["head"].set_data([], [])
                    del tracks[tid]

            # Capture
            if (frame_idx // DATA_FRAME_SKIP) % CAPTURE_INTERVAL == 0:
                writer.grab_frame()
                captured_frames += 1
                if captured_frames % 10 == 0:
                    sys.stdout.write(f"\rCaptured {captured_frames} GIF frames...")
                    sys.stdout.flush()

print(f"\nSuccess! Saved {OUTPUT_GIF_NAME}.")