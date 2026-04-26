import ijson
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button
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
DT = 0.033               
DATA_FRAME_SKIP = 1      

# -- Filter & Smoothing Toggles --
ENABLE_ALL_SMOOTHING = True      
ENABLE_POS_SMOOTHING = True      
ENABLE_HEADING_SMOOTHING = True  

GLOBAL_SMOOTHING = 0.1   
HEADING_SMOOTHING = 0.05 
JUMP_GATE_METERS = 2.0   

# -- Kinematic Bicycle Physics --
L = 2.7                  
MAX_STEER_DEG = 35.0     

# -- Visual / UI Controls --
DRAW_INTERVAL = 5        
LIFESPAN = 50            
MAX_SILENCE = 10         
DECAY_RATE = 1           
HEAD_MARKER_SIZE = 4     
COLOR_BY_CLASS = True    

ALLOWED_CLASSES = {"car", "truck", "bus", "pedestrian"}
VEHICLE_CLASSES = {"car", "truck", "bus"}

CLASS_COLOR_MAP = {
    "car": "cyan",
    "truck": "orange",
    "bus": "lime",
    "pedestrian": "magenta"
}

ID_PALETTE = [
    "tab:red", "tab:blue", "tab:green", "tab:purple", "tab:brown", 
    "tab:pink", "tab:olive", "tab:cyan", "gold", "magenta"
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
    try: return float(value)
    except (TypeError, ValueError): return default

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_track_color(tid, obj_class):
    if COLOR_BY_CLASS: return CLASS_COLOR_MAP.get(obj_class, "tab:blue")
    return ID_PALETTE[hash(tid) % len(ID_PALETTE)]

def load_svg_as_image(path, target_width, target_height):
    drawing = svg2rlg(path)
    scale_x = target_width / drawing.width
    scale_y = target_height / drawing.height
    drawing.width, drawing.height = target_width, target_height
    drawing.scale(scale_x, scale_y)
    img_data = renderPM.drawToString(drawing, fmt="PNG")
    return mpimg.imread(io.BytesIO(img_data))

# --- Line Intersection Math ---
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# ==========================================
# ===== SETUP FIGURE =====
# ==========================================

plt.ion()
fig, ax = plt.subplots(figsize=(12, 10))
fig_num = fig.number
plt.subplots_adjust(bottom=0.2)

sat_img = mpimg.imread(SAT_PATH)
h, w = sat_img.shape[:2]

try:
    svg_img = load_svg_as_image(SVG_PATH, w, h)
except Exception as e:
    svg_img = np.zeros((h, w, 4))

sat_layer = ax.imshow(sat_img, extent=[0, w, h, 0], alpha=0.0, zorder=1)
svg_layer = ax.imshow(svg_img, extent=[0, w, h, 0], alpha=1.0, zorder=2)

# ===== UI CONTROLS =====
ax_sat = plt.axes([0.15, 0.05, 0.5, 0.03])
ax_svg = plt.axes([0.15, 0.10, 0.5, 0.03])
s_sat = Slider(ax_sat, 'Sat Alpha', 0.0, 1.0, valinit=0.0)
s_svg = Slider(ax_svg, 'SVG Alpha', 0.0, 1.0, valinit=1.0)

def update_sliders(val):
    sat_layer.set_alpha(s_sat.val)
    svg_layer.set_alpha(s_svg.val)
    fig.canvas.draw_idle()

s_sat.on_changed(update_sliders)
s_svg.on_changed(update_sliders)

# ==========================================
# ===== DRAWING MODE (COUNTLINES) =====
# ==========================================
countlines = []
drawing_mode = True
current_line = None
start_pt = None
current_button = None

def on_press(event):
    if not drawing_mode or event.inaxes != ax: return
    # Only react to Left (1) or Right (3) click
    if event.button not in [1, 3]: return 
    
    global start_pt, current_line, current_button
    start_pt = (event.xdata, event.ydata)
    current_button = event.button
    
    ls = '-' if event.button == 1 else '--'
    current_line, = ax.plot([start_pt[0], start_pt[0]], [start_pt[1], start_pt[1]], 
                            color='gray', lw=2, zorder=5, linestyle=ls)
    fig.canvas.draw_idle()

def on_motion(event):
    if not drawing_mode or start_pt is None or event.inaxes != ax: return
    current_line.set_data([start_pt[0], event.xdata], [start_pt[1], event.ydata])
    fig.canvas.draw_idle()

def on_release(event):
    global start_pt, current_line, current_button, countlines
    if not drawing_mode or start_pt is None or event.inaxes != ax: return
    if event.button != current_button: return
    
    end_pt = (event.xdata, event.ydata)
    
    # Don't save if it's just a dot (click without drag)
    if np.linalg.norm(np.array(start_pt) - np.array(end_pt)) > 10:
        cid = len(countlines) + 1
        color = ID_PALETTE[cid % len(ID_PALETTE)]
        line_type = "vehicle" if event.button == 1 else "pedestrian"
        
        # Solidify line
        current_line.set_color(color)
        current_line.set_linewidth(3)
        
        # Create the inline text label
        label_str = "Vhl: 0" if line_type == "vehicle" else "Ppl: 0"
        txt_obj = ax.text(end_pt[0], end_pt[1], label_str, color=color, 
                          fontweight='bold', zorder=6,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color))
        
        countlines.append({
            "id": cid,
            "pts": (start_pt, end_pt),
            "type": line_type,
            "color": color,
            "count": 0,
            "text_obj": txt_obj,
            "crossed_tids": set()
        })
    else:
        current_line.remove() # discard small clicks
        
    start_pt = None
    current_line = None
    current_button = None
    fig.canvas.draw_idle()

cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
cid_release = fig.canvas.mpl_connect('button_release_event', on_release)

# Start Button
ax_start = plt.axes([0.75, 0.05, 0.15, 0.08])
btn_start = Button(ax_start, 'START', color='lightgreen', hovercolor='palegreen')

def start_sim(event):
    global drawing_mode
    drawing_mode = False
    btn_start.color = 'lightgray'
    btn_start.label.set_text('RUNNING')
    
    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_motion)
    fig.canvas.mpl_disconnect(cid_release)

btn_start.on_clicked(start_sim)

print("--- DRAWING MODE ---")
print("LEFT CLICK + Drag: Draw Vehicle Line (Solid)")
print("RIGHT CLICK + Drag: Draw Pedestrian Line (Dashed)")
print("Press the 'START' button when you are ready.")

# Wait for user
while drawing_mode:
    if not plt.fignum_exists(fig_num):
        sys.exit(0)
    plt.pause(0.1)

# ==========================================
# ===== MAIN LOOP =====
# ==========================================

tracks = {}
effective_dt = DT * DATA_FRAME_SKIP

try:
    with open(DATA_PATH, "rb") as f:
        frames = ijson.items(f, "frames.item")
        
        for frame_idx, frame in enumerate(frames):
            
            if not plt.fignum_exists(fig_num):
                break
                
            if frame_idx % DATA_FRAME_SKIP != 0:
                continue

            seen_tids = set()
            stats_updated = False 

            for obj in frame.get("objects", []):
                obj_class = obj.get("class", "").lower()
                if obj_class not in ALLOWED_CLASSES:
                    continue

                tid = obj["tracked_id"]
                seen_tids.add(tid)
                
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
                        "state": np.array([z_x, z_y, z_theta, z_v]),
                        "x": [z_x], "y": [z_y],
                        "inactive_frames": 0, "class": obj_class
                    }
                    tracks[tid]["line"].set_data(tracks[tid]["x"], tracks[tid]["y"])
                    tracks[tid]["head"].set_data([z_x], [z_y])
                    continue

                # --- KINEMATIC BICYCLE UPDATE ---
                curr_s = tracks[tid]["state"]
                c_x, c_y, c_theta, c_v = curr_s

                dist_to_cam = np.linalg.norm(z_pos - CAM_SAT_POS)
                dynamic_gain = GLOBAL_SMOOTHING * np.clip(500/max(dist_to_cam, 1), 0.2, 1.0)
                
                if np.linalg.norm(z_pos - np.array([c_x, c_y])) / PX_PER_METER > JUMP_GATE_METERS:
                    dynamic_gain *= 0.1

                if ENABLE_ALL_SMOOTHING:
                    theta_diff = normalize_angle(z_theta - c_theta)
                    steer_req = np.arctan((theta_diff / effective_dt) * (L / c_v)) if abs(c_v) > 0.1 else 0.0
                    
                    max_steer_rad = np.radians(MAX_STEER_DEG)
                    steer = np.clip(steer_req, -max_steer_rad, max_steer_rad)

                    pred_theta = normalize_angle(c_theta + (c_v / L) * np.tan(steer) * effective_dt)
                    pred_x = c_x + c_v * np.cos(pred_theta) * effective_dt
                    pred_y = c_y + c_v * np.sin(pred_theta) * effective_dt

                    smooth_x = pred_x + dynamic_gain * (z_x - pred_x) if ENABLE_POS_SMOOTHING else z_x
                    smooth_y = pred_y + dynamic_gain * (z_y - pred_y) if ENABLE_POS_SMOOTHING else z_y
                    smooth_theta = pred_theta + HEADING_SMOOTHING * normalize_angle(z_theta - pred_theta) if ENABLE_HEADING_SMOOTHING else z_theta
                else:
                    smooth_x, smooth_y, smooth_theta = z_x, z_y, z_theta

                # --- COUNTLINE INTERSECTION CHECK ---
                prev_x, prev_y = tracks[tid]["x"][-1], tracks[tid]["y"][-1]
                for cl in countlines:
                    if tid not in cl["crossed_tids"]:
                        if intersect((prev_x, prev_y), (smooth_x, smooth_y), cl["pts"][0], cl["pts"][1]):
                            
                            is_vehicle = obj_class in VEHICLE_CLASSES
                            is_pedestrian = obj_class == "pedestrian"
                            
                            if (cl["type"] == "vehicle" and is_vehicle) or (cl["type"] == "pedestrian" and is_pedestrian):
                                cl["crossed_tids"].add(tid)
                                cl["count"] += 1
                                
                                # Update the text object floating next to the line
                                prefix = "Vhl" if cl["type"] == "vehicle" else "Ppl"
                                cl["text_obj"].set_text(f"{prefix}: {cl['count']}")
                                stats_updated = True

                # Save Data
                tracks[tid]["state"] = np.array([smooth_x, smooth_y, smooth_theta, z_v])
                tracks[tid]["inactive_frames"] = 0
                tracks[tid]["x"].append(smooth_x)
                tracks[tid]["y"].append(smooth_y)
                
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

                if len(data["x"]) > 0: data["x"] = data["x"][DECAY_RATE:]
                if len(data["y"]) > 0: data["y"] = data["y"][DECAY_RATE:]

                remaining_ratio = len(data["x"]) / max(1, LIFESPAN)
                alpha = max(0.1, 0.8 * remaining_ratio)
                
                if data["x"] and data["y"]:
                    data["line"].set_alpha(alpha)
                    data["line"].set_data(data["x"], data["y"])
                    data["head"].set_data([data["x"][-1]], [data["y"][-1]])
                    data["head"].set_alpha(alpha)
                else:
                    data["line"].set_data([], [])
                    data["head"].set_data([], [])
                    del tracks[tid]

            # --- RENDER ---
            if (frame_idx // DATA_FRAME_SKIP) % DRAW_INTERVAL == 0:
                ax.set_title(f"Shinjuku 1 | Frame: {frame_idx}")
                plt.draw()
                plt.pause(0.001)

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