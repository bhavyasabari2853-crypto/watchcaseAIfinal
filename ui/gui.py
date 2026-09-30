import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import tkinter.scrolledtext as scrolledtext
import shutil
import os
import sys
import json
import threading
import tempfile
import cv2
from datetime import datetime

# Define paths and import system
file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ReferenceManager
from src.socket_server import start_server
from src.reference_manager import ReferenceManager
from src.webcam_infer import detect_on_frame, write_error_status
from ultralytics import YOLO

# Single Path Manager
def data_path(relative_path):
    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = project_root
    return os.path.join(base_path, relative_path)

REF_DIR = data_path("references")
MODEL_PATH = data_path("model/best.pt")
DETECTION_FILE = data_path("latest_detection.json")

os.makedirs(REF_DIR, exist_ok=True)

model = None
ref_mgr = ReferenceManager(ref_dir=REF_DIR)
cap = None
camera_lock = threading.Lock()

_detection_window_ref = None

# Safe model loading function
def load_model():
    global model
    if model is not None:
        return True

    if not os.path.exists(MODEL_PATH):
        messagebox.showerror(
            "Model file not found",
            f"Model file not found\n\nExpected location:\n{MODEL_PATH}"
        )
        return False

    try:
        log_message("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        log_message("YOLO model loaded.")
        return True
    except Exception as e:
        messagebox.showerror(
            "Error loading model",
            f"Failed to load YOLO model:\n{e}\n\nExpected location:\n{MODEL_PATH}"
        )
        return False

def init_camera():
    global cap
    if cap is None or not cap.isOpened():
        log_message("Opening camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            log_message("Error: Cannot open camera")
        else:
            log_message("Camera opened successfully.")

# Startup warning check
def check_model_on_startup():
    if not os.path.exists(MODEL_PATH):
        messagebox.showwarning(
            "Model file missing",
            f"Model file not found at:\n{MODEL_PATH}\n\nYou can still configure references, but detection will not run until the model file is placed."
        )

# --------------------------
# Upload Reference
# --------------------------
def upload_reference():
    case_name = case_name_entry.get().strip()
    if not case_name:
        messagebox.showerror("Error", "Enter Case Name")
        return

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    try:
        ref_mgr.add(file_path, case_name)
        messagebox.showinfo("Success", f"Saved and activated: {case_name}")
        update_active_ref_label()
        case_name_entry.delete(0, tk.END)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to add reference: {e}")

# --------------------------
# Capture Reference from Camera
# --------------------------
def capture_reference():
    from PIL import Image, ImageTk

    capture_result = {"path": None, "done": False, "running": True}

    global cap
    if cap is None or not cap.isOpened():
        init_camera()
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        capture_result["done"] = True
        return

    preview = tk.Toplevel(root)
    preview.title("Capture Reference")
    preview.geometry("660x560")
    preview.resizable(False, False)

    inst = tk.Label(preview, text="Press C to Capture | Q to Cancel", font=("Arial", 12, "bold"))
    inst.pack(pady=5)

    video_label = tk.Label(preview)
    video_label.pack()

    def close_cleanup():
        if capture_result["running"]:
            capture_result["running"] = False
            try:
                preview.destroy()
            except tk.TclError:
                pass
            capture_result["done"] = True

    preview.protocol("WM_DELETE_WINDOW", close_cleanup)

    def update_frame():
        if not capture_result["running"]:
            return
        with camera_lock:
            ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
        preview.after(30, update_frame)

    def on_key(event):
        if not capture_result["running"]:
            return
        if event.char == 'c':
            log_message("Key pressed: C")
            with camera_lock:
                ret, frame = cap.read()
            if ret:
                temp_path = os.path.join(tempfile.gettempdir(), "watchcase_capture.jpg")
                cv2.imwrite(temp_path, frame)
                log_message("Capture Saved")
                capture_result["path"] = temp_path
            capture_result["running"] = False
            preview.destroy()
            capture_result["done"] = True
        elif event.char == 'q':
            log_message("Key pressed: Q")
            capture_result["running"] = False
            preview.destroy()
            capture_result["done"] = True

    preview.bind('<KeyPress>', on_key)
    preview.focus_set()

    update_frame()

    def _check_done():
        if capture_result["done"]:
            path = capture_result["path"]
            if path:
                name = simpledialog.askstring(
                    "Reference Name",
                    "Enter name for the captured reference:",
                    parent=root
                )
                if name and name.strip():
                    try:
                        ref_mgr.add(path, name.strip())
                        messagebox.showinfo(
                            "Success",
                            f"Reference '{name.strip()}' captured and activated"
                        )
                        update_active_ref_label()
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to save reference: {e}")
                    finally:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
        else:
            root.after(200, _check_done)

    root.after(200, _check_done)

# --------------------------
# Detection Window (Tkinter-based, no OpenCV HighGUI)
# --------------------------
def open_detection_window():
    global _detection_window_ref, cap, model
    from PIL import Image, ImageTk

    if not load_model():
        return

    if _detection_window_ref is not None:
        try:
            _detection_window_ref.lift()
            _detection_window_ref.focus_set()
            return
        except tk.TclError:
            _detection_window_ref = None

    if cap is None or not cap.isOpened():
        init_camera()
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        return

    state = {"running": True, "detecting": False, "showing_result": False}

    win = tk.Toplevel(root)
    win.title("Watch Case Detection")
    win.geometry("700x600")
    _detection_window_ref = win

    status_lbl = tk.Label(win, text="Press S to Detect | Q to Quit", font=("Arial", 12, "bold"))
    status_lbl.pack(pady=5)

    video_label = tk.Label(win)
    video_label.pack()

    win.status_lbl = status_lbl
    win.video_label = video_label
    win.state = state

    def cleanup():
        global _detection_window_ref
        if not state["running"]:
            return
        state["running"] = False
        _detection_window_ref = None
        try:
            win.destroy()
        except tk.TclError:
            pass

    win.protocol("WM_DELETE_WINDOW", cleanup)

    def update_frame():
        if not state["running"]:
            return

        if state["showing_result"]:
            win.after(30, update_frame)
            return

        with camera_lock:
            ret, frame = cap.read()

        if ret:
            display_frame = frame.copy()

            if state["detecting"]:
                cv2.putText(
                    display_frame,
                    "DETECTING...",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(img)

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

        win.after(30, update_frame)

    def do_detect():
        if state["detecting"] or not state["running"]:
            return
        state["detecting"] = True      
        state["showing_result"] = False
        status_lbl.config(text="Detecting...")
        log_message("Detection Started")

        with camera_lock:
            ret, frame = cap.read()
        if not ret:
            state["detecting"] = False
            status_lbl.config(text="Camera read failed")
            return

        frame_copy = frame.copy()

        def _detect_worker():
            try:
                success, result_img = detect_on_frame(model, ref_mgr, frame_copy, DETECTION_FILE)
                win.after(0, _on_result, success, result_img)
            except Exception as e:
                win.after(0, _on_error, str(e))

        threading.Thread(target=_detect_worker, daemon=True).start()

    def _on_result(success, result_img):
        state["detecting"] = False
        log_message("Detection Completed")
        if success:
            state["showing_result"] = True
            status_lbl.config(text="Detection completed. Showing overlay...")
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(result_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
        else:
            status_lbl.config(text="No object detected. Press S to try again.")

    def _on_error(err):
        state["detecting"] = False
        log_message(f"Detection error: {err}")
        status_lbl.config(text=f"Error: {err}")

    def on_key(event):
        if not state["running"]:
            return

        key = event.char.lower()

        if key == 'q':
            log_message("Key pressed: Q")
            cleanup()

        elif key == 's':
            log_message("Key pressed: S")

            if state["showing_result"]:
                state["showing_result"] = False
                status_lbl.config(text="Press S to Detect | Q to Quit")

            do_detect()

    win.focus_set()
    win.bind('<KeyPress>', on_key)
    update_frame()

def display_overlay_in_gui(result_img):
    global _detection_window_ref
    from PIL import Image, ImageTk
    if _detection_window_ref is None:
        open_detection_window()
    if _detection_window_ref is not None:
        try:
            win = _detection_window_ref
            win.state["showing_result"] = True
            win.state["detecting"] = False
            win.status_lbl.config(text="Detection completed. Showing overlay...")
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(result_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            win.video_label.imgtk = imgtk
            win.video_label.config(image=imgtk)
        except Exception as e:
            log_message(f"Error updating overlay in GUI: {e}")

def display_no_detection_in_gui():
    global _detection_window_ref
    if _detection_window_ref is None:
        open_detection_window()
    if _detection_window_ref is not None:
        try:
            win = _detection_window_ref
            win.state["showing_result"] = False
            win.state["detecting"] = False
            win.status_lbl.config(text="No object detected. Waiting for next trigger...")
        except Exception as e:
            log_message(f"Error updating GUI status: {e}")

def reset_preview_in_gui():
    global _detection_window_ref
    if _detection_window_ref is not None:
        try:
            win = _detection_window_ref
            win.state["showing_result"] = False
            win.state["detecting"] = True
            win.status_lbl.config(text="Refreshing preview...")
        except Exception as e:
            log_message(f"Error resetting preview in GUI: {e}")

def on_socket_trigger():
    global model, ref_mgr, cap, DETECTION_FILE
    import time
    log_message("Detection Started")

    # 1. Reset/Refresh GUI to live preview state before camera grab/detection
    root.after(0, reset_preview_in_gui)

    # Sleep to allow GUI thread to read and display a few fresh frames from the webcam,
    # thereby flushing the camera hardware buffer and displaying the new camera frame.
    time.sleep(0.2)

    if cap is None or not cap.isOpened():
        log_message("Error: Camera not open")
        write_error_status(DETECTION_FILE, "Camera not open")
        log_message("Detection Completed")
        return json.dumps({"detection_status": "Camera not open", "detections": []})
    if model is None:
        log_message("Error: YOLO model not loaded")
        write_error_status(DETECTION_FILE, "YOLO model not loaded")
        log_message("Detection Completed")
        return json.dumps({"detection_status": "YOLO model not loaded", "detections": []})

    with camera_lock:
        for _ in range(5):
            cap.grab()
        ret, frame = cap.retrieve()

    if not ret:
        log_message("Error: Camera capture failed")
        write_error_status(DETECTION_FILE, "Camera capture failed")
        log_message("Detection Completed")
        return json.dumps({"detection_status": "Camera capture failed", "detections": []})

    frame_copy = frame.copy()
    try:
        success, result_img = detect_on_frame(model, ref_mgr, frame_copy, DETECTION_FILE)
        log_message("Detection Completed")
        if success:
            root.after(0, display_overlay_in_gui, result_img)
        else:
            root.after(0, display_no_detection_in_gui)
    except Exception as e:
        log_message(f"Detection error: {e}")
        write_error_status(DETECTION_FILE, f"Detection error: {e}")
        log_message("Detection Completed")

    # Overlay stays on screen forever until next trigger

    try:
        if os.path.exists(DETECTION_FILE):
            with open(DETECTION_FILE, "r") as f:
                return f.read()
    except Exception as e:
        log_message(f"Error reading JSON: {e}")
    return json.dumps({"detection_status": "Error reading JSON", "detections": []})

# --------------------------
# Load Latest Detection
# --------------------------
def load_latest_detection():
    try:
        if os.path.exists(DETECTION_FILE):
            with open(DETECTION_FILE, "r") as f:
                data = json.load(f)
            status_var.set(data.get("detection_status", ""))

            # Reset variables
            obj_name_var1.set("")
            cx_var1.set("")
            cy_var1.set("")
            angle_var1.set("")

            obj_name_var2.set("")
            cx_var2.set("")
            cy_var2.set("")
            angle_var2.set("")

            detections = data.get("detections", [])
            # Fallback for old format
            if not detections and "object_name" in data:
                detections = [data]

            if len(detections) > 0:
                d = detections[0]
                obj_name_var1.set(d.get("object_name", ""))
                cx_val = d.get("centroid_x", "")
                cy_val = d.get("centroid_y", "")
                ang_val = d.get("angle", "")
                cx_var1.set(f"{cx_val:.2f}" if isinstance(cx_val, float) else str(cx_val))
                cy_var1.set(f"{cy_val:.2f}" if isinstance(cy_val, float) else str(cy_val))
                angle_var1.set(f"{ang_val:.2f}" if isinstance(ang_val, float) else str(ang_val))

            if len(detections) > 1:
                d = detections[1]
                obj_name_var2.set(d.get("object_name", ""))
                cx_val = d.get("centroid_x", "")
                cy_val = d.get("centroid_y", "")
                ang_val = d.get("angle", "")
                cx_var2.set(f"{cx_val:.2f}" if isinstance(cx_val, float) else str(cx_val))
                cy_var2.set(f"{cy_val:.2f}" if isinstance(cy_val, float) else str(cy_val))
                angle_var2.set(f"{ang_val:.2f}" if isinstance(ang_val, float) else str(ang_val))
    except Exception as e:
        log_message(f"Error reading detection JSON: {e}")

    root.after(1000, load_latest_detection)

# --------------------------
# GUI
# --------------------------
root = tk.Tk()
root.title("Watch Case Recognition System")
root.geometry("720x820")

title = tk.Label(
    root,
    text="Watch Case Recognition System",
    font=("Arial", 16, "bold")
)
title.pack(pady=15)

tk.Label(root, text="Case Name:").pack()

case_name_entry = tk.Entry(root, width=30)
case_name_entry.pack(pady=5)

# Active Reference Status label (added to inform users of the active reference state)
active_ref_var = tk.StringVar(value="None")
def update_active_ref_label():
    name = ref_mgr.get_active_name()
    active_ref_var.set(name if name else "None")

active_ref_frame = tk.Frame(root)
active_ref_frame.pack(pady=2)
tk.Label(active_ref_frame, text="Active Reference: ").pack(side="left")
tk.Label(active_ref_frame, textvariable=active_ref_var, font=("Arial", 9, "bold")).pack(side="left")

tk.Button(
    root,
    text="Upload & Save Reference",
    command=upload_reference,
    width=30
).pack(pady=10)

tk.Button(
    root,
    text="Capture Reference",
    command=capture_reference,
    width=30
).pack(pady=10)

tk.Button(
    root,
    text="Start Detection",
    command=open_detection_window,
    width=30
).pack(pady=10)

# Latest Detection Panel
detection_frame = tk.LabelFrame(
    root,
    text="Latest Detections",
    font=("Arial", 12, "bold"),
    padx=15,
    pady=15
)
detection_frame.pack(pady=10, padx=20, fill="x")

status_var = tk.StringVar(value="")
obj_name_var1 = tk.StringVar(value="")
cx_var1 = tk.StringVar(value="")
cy_var1 = tk.StringVar(value="")
angle_var1 = tk.StringVar(value="")

obj_name_var2 = tk.StringVar(value="")
cx_var2 = tk.StringVar(value="")
cy_var2 = tk.StringVar(value="")
angle_var2 = tk.StringVar(value="")

# Detection Status row
tk.Label(
    detection_frame,
    text="Detection Status:",
    font=("Arial", 10, "bold"),
    anchor="w"
).grid(row=0, column=0, sticky="w", pady=4, padx=(0, 10))

tk.Label(
    detection_frame,
    textvariable=status_var,
    font=("Arial", 10, "bold"),
    anchor="w",
    fg="#0066cc"
).grid(row=0, column=1, columnspan=3, sticky="w", pady=4)

# Detection 1 header and info
tk.Label(
    detection_frame,
    text="Detection 1",
    font=("Arial", 11, "bold", "underline"),
    anchor="w",
    fg="#008000"
).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 4))

info_data1 = [
    ("Object Name:", obj_name_var1, 2),
    ("Centroid X:", cx_var1, 3),
    ("Centroid Y:", cy_var1, 4),
    ("Angle (deg):", angle_var1, 5),
]

for label_text, var, row in info_data1:
    tk.Label(
        detection_frame,
        text=label_text,
        font=("Arial", 10, "bold"),
        anchor="w"
    ).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 10))

    tk.Label(
        detection_frame,
        textvariable=var,
        font=("Arial", 10),
        anchor="w",
        fg="#333333"
    ).grid(row=row, column=1, sticky="w", pady=2)

# Detection 2 header and info
tk.Label(
    detection_frame,
    text="Detection 2",
    font=("Arial", 11, "bold", "underline"),
    anchor="w",
    fg="#008000"
).grid(row=1, column=2, columnspan=2, sticky="w", pady=(8, 4), padx=(40, 0))

info_data2 = [
    ("Object Name:", obj_name_var2, 2),
    ("Centroid X:", cx_var2, 3),
    ("Centroid Y:", cy_var2, 4),
    ("Angle (deg):", angle_var2, 5),
]

for label_text, var, row in info_data2:
    tk.Label(
        detection_frame,
        text=label_text,
        font=("Arial", 10, "bold"),
        anchor="w"
    ).grid(row=row, column=2, sticky="w", pady=2, padx=(40, 10))

    tk.Label(
        detection_frame,
        textvariable=var,
        font=("Arial", 10),
        anchor="w",
        fg="#333333"
    ).grid(row=row, column=3, sticky="w", pady=2)

# --------------------------
# Communication Status Panel
# --------------------------
comm_frame = tk.LabelFrame(
    root,
    text="Communication",
    font=("Arial", 12, "bold"),
    padx=15,
    pady=10
)
comm_frame.pack(pady=10, padx=20, fill="x")

server_status_var = tk.StringVar(value="Starting...")
client_ip_var = tk.StringVar(value="None")
last_trigger_var = tk.StringVar(value="None")
last_data_var = tk.StringVar(value="None")

status_rows = [
    ("Server Status:", server_status_var, 0),
    ("Connected Client:", client_ip_var, 1),
    ("Last Trigger Time:", last_trigger_var, 2),
    ("Last Data Sent:", last_data_var, 3),
]

for label_text, var, row in status_rows:
    tk.Label(
        comm_frame,
        text=label_text,
        font=("Arial", 10, "bold"),
        anchor="w"
    ).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 10))

    tk.Label(
        comm_frame,
        textvariable=var,
        font=("Arial", 10),
        anchor="w",
        fg="#333333"
    ).grid(row=row, column=1, sticky="w", pady=2)

# --------------------------
# Communication Log Panel
# --------------------------
log_frame = tk.LabelFrame(
    root,
    text="Communication Log",
    font=("Arial", 12, "bold"),
    padx=10,
    pady=10
)
log_frame.pack(pady=10, padx=20, fill="both", expand=True)

log_text = scrolledtext.ScrolledText(
    log_frame,
    height=8,
    width=70,
    state=tk.DISABLED,
    wrap=tk.WORD,
    font=("Consolas", 9)
)
log_text.pack(fill="both", expand=True)

# --------------------------
# Callback functions for socket server
# --------------------------
def log_message(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    root.after(0, _append_log, formatted + "\n")

def _append_log(text):
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, text)
    log_text.see(tk.END)
    log_text.config(state=tk.DISABLED)

def update_status(key, value):
    root.after(0, _set_status, key, value)

def _set_status(key, value):
    if key == "server_status":
        server_status_var.set(value)
    elif key == "client_ip":
        client_ip_var.set(value)
    elif key == "last_trigger":
        last_trigger_var.set(value)
    elif key == "last_data":
        last_data_var.set(value)

# --------------------------
# Start socket server in daemon thread with callbacks
# --------------------------
server_thread = threading.Thread(
    target=start_server,
    args=(log_message, update_status, DETECTION_FILE, on_socket_trigger),
    daemon=True
)
server_thread.start()

# Initialization
update_active_ref_label()
load_latest_detection()
load_model()
init_camera()
root.after(100, check_model_on_startup)
root.after(150, open_detection_window)
root.mainloop()
