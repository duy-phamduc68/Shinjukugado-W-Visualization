import ijson
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button
from matplotlib.animation import PillowWriter
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import io
import os
import sys

# ==========================================
# ===== GIF EXPORT CONFIGURATION =====
# ==========================================
OUTPUT_GIF_NAME = "shinjuku_countlines_optimized.gif"

# -- Strict Web-Optimization Settings --
GIF_FPS = 10             
GIF_DPI = 60             
DATA_FRAME_SKIP = 2      
CAPTURE_INTERVAL = 24    

# Visual Overrides for GIF Output
STATIC_SAT_ALPHA = 0.15
STATIC_SVG_ALPHA = 1.0

# ==========================================
# ===== PROJECT & PHYSICS CONFIG =====
# ==========================================
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

LIFESPAN = 30            
MAX_SILENCE = 5          
DECAY_RATE = 2           
HEAD_MARKER_SIZE = 3     

ALLOWED_CLASSES = {"car", "truck", "bus", "pedestrian"}
VEHICLE_CLASSES = {"car", "truck", "bus"}

CLASS_COLOR_MAP = {"car": "cyan", "truck": "orange", "bus": "lime", "pedestrian": "magenta"}
ID_PALETTE = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "gold"]

# ==========================================
# ===== HELPERS & DATA LOADING =====
# ==========================================

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

# --- Line Intersection Math ---
def ccw(A, B, C): return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
def intersect(A, B, C, D): return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

sat_img = mpimg.imread(SAT_PATH)
h, w = sat_img.shape[:2]
try: svg_img = load_svg_as_image(SVG_PATH, w, h)
except: svg_img = np.zeros((h, w, 4))

# ==========================================
# ===== PHASE 1: UI DRAWING MODE =====
# ==========================================
plt.ion()
fig_ui, ax_ui = plt.subplots(figsize=(12, 10))
fig_num = fig_ui.number
plt.subplots_adjust(bottom=0.2)

sat_layer = ax_ui.imshow(sat_img, extent=[0, w, h, 0], alpha=0.0, zorder=1)
svg_layer = ax_ui.imshow(svg_img, extent=[0, w, h, 0], alpha=1.0, zorder=2)
ax_ui.set_title("DRAWING MODE: Draw lines, then press START")

ax_sat = plt.axes([0.15, 0.05, 0.5, 0.03])
ax_svg = plt.axes([0.15, 0.10, 0.5, 0.03])
s_sat = Slider(ax_sat, 'Sat Alpha', 0.0, 1.0, valinit=0.0)
s_svg = Slider(ax_svg, 'SVG Alpha', 0.0, 1.0, valinit=1.0)

def update_sliders(val):
    sat_layer.set_alpha(s_sat.val)
    svg_layer.set_alpha(s_svg.val)
    fig_ui.canvas.draw_idle()

s_sat.on_changed(update_sliders)
s_svg.on_changed(update_sliders)

countlines = []
drawing_mode = True
current_line = None
start_pt = None
current_button = None

def on_press(event):
    if not drawing_mode or event.inaxes != ax_ui: return
    if event.button not in [1, 3]: return 
    global start_pt, current_line, current_button
    start_pt = (event.xdata, event.ydata)
    current_button = event.button
    ls = '-' if event.button == 1 else '--'
    current_line, = ax_ui.plot([start_pt[0], start_pt[0]], [start_pt[1], start_pt[1]], color='gray', lw=2, zorder=5, linestyle=ls)
    fig_ui.canvas.draw_idle()

def on_motion(event):
    if not drawing_mode or start_pt is None or event.inaxes != ax_ui: return
    current_line.set_data([start_pt[0], event.xdata], [start_pt[1], event.ydata])
    fig_ui.canvas.draw_idle()

def on_release(event):
    global start_pt, current_line, current_button, countlines
    if not drawing_mode or start_pt is None or event.inaxes != ax_ui: return
    if event.button != current_button: return
    
    end_pt = (event.xdata, event.ydata)
    
    if np.linalg.norm(np.array(start_pt) - np.array(end_pt)) > 10:
        cid = len(countlines) + 1
        color = ID_PALETTE[cid % len(ID_PALETTE)]
        line_type = "vehicle" if event.button == 1 else "pedestrian"
        
        current_line.set_color(color)
        current_line.set_linewidth(3)
        label_str = "Vhl: 0" if line_type == "vehicle" else "Ppl: 0"
        ax_ui.text(end_pt[0], end_pt[1], label_str, color=color, fontweight='bold', zorder=6,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color))
        
        countlines.append({
            "id": cid, "pts": (start_pt, end_pt), "type": line_type, "color": color, 
            "count": 0, "crossed_tids": set()
        })
    else:
        current_line.remove()
        
    start_pt, current_line, current_button = None, None, None
    fig_ui.canvas.draw_idle()

cid_press = fig_ui.canvas.mpl_connect('button_press_event', on_press)
cid_motion = fig_ui.canvas.mpl_connect('motion_notify_event', on_motion)
cid_release = fig_ui.canvas.mpl_connect('button_release_event', on_release)

ax_start = plt.axes([0.75, 0.05, 0.15, 0.08])
btn_start = Button(ax_start, 'START', color='lightgreen', hovercolor='palegreen')

def start_sim(event):
    global drawing_mode
    drawing_mode = False

btn_start.on_clicked(start_sim)

print("--- DRAWING MODE ---")
print("LEFT CLICK + Drag: Draw Vehicle Line (Solid)")
print("RIGHT CLICK + Drag: Draw Pedestrian Line (Dashed)")
print("Press 'START' when finished to render GIF.")

while drawing_mode:
    if not plt.fignum_exists(fig_num): sys.exit(0)
    plt.pause(0.1)

# Clean up UI
plt.close('all')
plt.ioff()

# ==========================================
# ===== PHASE 2: HEADLESS GIF RENDER =====
# ==========================================
print(f"\nUI closed. Building strict GIF environment for {OUTPUT_GIF_NAME}...")

fig_gif, ax_gif = plt.subplots(figsize=(8, 6))
ax_gif.imshow(sat_img, extent=[0, w, h, 0], alpha=STATIC_SAT_ALPHA, zorder=1)
ax_gif.imshow(svg_img, extent=[0, w, h, 0], alpha=STATIC_SVG_ALPHA, zorder=2)
ax_gif.set_xticks([]); ax_gif.set_yticks([])
ax_gif.set_title("Traffic Countlines", pad=10)
fig_gif.tight_layout()

# Reconstruct Countlines on new Figure
for cl in countlines:
    ls = '-' if cl["type"] == "vehicle" else '--'
    ax_gif.plot([cl["pts"][0][0], cl["pts"][1][0]], [cl["pts"][0][1], cl["pts"][1][1]], 
                color=cl["color"], lw=3, zorder=5, linestyle=ls)
    
    prefix = "Vhl" if cl["type"] == "vehicle" else "Ppl"
    cl["text_obj"] = ax_gif.text(cl["pts"][1][0], cl["pts"][1][1], f"{prefix}: 0", 
                                 color=cl["color"], fontweight='bold', zorder=6,
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=cl["color"]))

tracks = {}
effective_dt = DT * DATA_FRAME_SKIP
writer = PillowWriter(fps=GIF_FPS)

print("Starting headless simulation and frame capture...")

with writer.saving(fig_gif, OUTPUT_GIF_NAME, dpi=GIF_DPI):
    with open(DATA_PATH, "rb") as f:
        frames = ijson.items(f, "frames.item")
        captured_frames = 0
        
        for frame_idx, frame in enumerate(frames):
            if frame_idx % DATA_FRAME_SKIP != 0: continue

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
                    line, = ax_gif.plot([], [], linewidth=1.5, alpha=0.8, zorder=3, color=color)
                    head, = ax_gif.plot([], [], marker='o', markersize=HEAD_MARKER_SIZE, color=color, zorder=4)
                    tracks[tid] = {
                        "line": line, "head": head, "state": np.array([z_x, z_y, z_theta, z_v]), 
                        "x": [z_x], "y": [z_y], "inactive_frames": 0, "class": obj_class
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

                # --- COUNTLINE CROSSING ---
                prev_x, prev_y = tracks[tid]["x"][-1], tracks[tid]["y"][-1]
                for cl in countlines:
                    if tid not in cl["crossed_tids"]:
                        if intersect((prev_x, prev_y), (smooth_x, smooth_y), cl["pts"][0], cl["pts"][1]):
                            is_vehicle = obj_class in VEHICLE_CLASSES
                            is_ped = obj_class == "pedestrian"
                            if (cl["type"] == "vehicle" and is_vehicle) or (cl["type"] == "pedestrian" and is_ped):
                                cl["crossed_tids"].add(tid)
                                cl["count"] += 1
                                prefix = "Vhl" if cl["type"] == "vehicle" else "Ppl"
                                cl["text_obj"].set_text(f"{prefix}: {cl['count']}")

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

            # Capture Frame
            if (frame_idx // DATA_FRAME_SKIP) % CAPTURE_INTERVAL == 0:
                writer.grab_frame()
                captured_frames += 1
                if captured_frames % 5 == 0:
                    sys.stdout.write(f"\rCaptured {captured_frames} GIF frames...")
                    sys.stdout.flush()

print(f"\nSuccess! Saved {OUTPUT_GIF_NAME}.")