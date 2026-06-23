import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import shutil
import os
import sys
import json
import threading
import tempfile
import cv2


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def data_path(relative_path):
    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


if getattr(sys, "frozen", False):
    _BASE = sys._MEIPASS
else:
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from ultralytics import YOLO
from src.reference_manager import ReferenceManager

REF_DIR = data_path("references")
MODEL_PATH = resource_path("model/best.pt")
DETECTION_FILE = data_path("latest_detection.json")

os.makedirs(REF_DIR, exist_ok=True)

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded.")

ref_mgr = ReferenceManager(ref_dir=REF_DIR)

detection_thread = None
detection_stop = threading.Event()


# --------------------------
# Upload Reference
# --------------------------
def upload_reference():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    case_name = case_name_entry.get().strip()
    if not case_name:
        messagebox.showerror("Error", "Enter Case Name")
        return

    save_path = os.path.join(REF_DIR, f"{case_name}.jpg")
    shutil.copy(file_path, save_path)
    ref_mgr.activate(case_name)

    messagebox.showinfo("Success", f"Saved and activated: {case_name}")


# --------------------------
# Capture Reference from Camera
# --------------------------
def capture_reference():
    capture_result = {"path": None, "done": False}

    def _capture():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            root.after(0, lambda: messagebox.showerror("Error", "Cannot open camera"))
            capture_result["done"] = True
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            cv2.putText(
                display,
                "Press C to Capture | Q to Cancel",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.imshow("Capture Reference", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                temp_path = os.path.join(tempfile.gettempdir(), "watchcase_capture.jpg")
                cv2.imwrite(temp_path, frame)
                capture_result["path"] = temp_path
                break

        cap.release()
        cv2.destroyAllWindows()
        capture_result["done"] = True

    threading.Thread(target=_capture, daemon=True).start()

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
                    ref_mgr.add(path, name.strip())
                    messagebox.showinfo(
                        "Success",
                        f"Reference '{name.strip()}' captured and activated"
                    )
        else:
            root.after(200, _check_done)

    root.after(200, _check_done)


# --------------------------
# Start Detection
# --------------------------
def start_detection():
    global detection_thread
    if detection_thread is not None and detection_thread.is_alive():
        messagebox.showinfo("Info", "Detection is already running")
        return

    from src.webcam_infer import run_detection_loop

    detection_stop.clear()

    detection_thread = threading.Thread(
        target=run_detection_loop,
        args=(model, ref_mgr, DETECTION_FILE),
        daemon=True
    )
    detection_thread.start()


# --------------------------
# Load Latest Detection
# --------------------------
def load_latest_detection():
    try:
        with open(DETECTION_FILE, "r") as f:
            data = json.load(f)
        obj_name_var.set(data.get("object_name", ""))
        cx_var.set(str(data.get("centroid_x", "")))
        cy_var.set(str(data.get("centroid_y", "")))
        angle_var.set(str(data.get("angle", "")))
        status_var.set(data.get("detection_status", ""))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass
    root.after(1000, load_latest_detection)


# --------------------------
# GUI
# --------------------------
root = tk.Tk()
root.title("Watch Case Recognition System")
root.geometry("550x550")

title = tk.Label(
    root,
    text="Watch Case Recognition System",
    font=("Arial", 16, "bold")
)
title.pack(pady=15)

tk.Label(root, text="Case Name:").pack()

case_name_entry = tk.Entry(root, width=30)
case_name_entry.pack(pady=5)

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
    command=start_detection,
    width=30
).pack(pady=10)

# Latest Detection Panel
detection_frame = tk.LabelFrame(
    root,
    text="Latest Detection",
    font=("Arial", 12, "bold"),
    padx=15,
    pady=15
)
detection_frame.pack(pady=20, padx=20, fill="both")

obj_name_var = tk.StringVar(value="")
cx_var = tk.StringVar(value="")
cy_var = tk.StringVar(value="")
angle_var = tk.StringVar(value="")
status_var = tk.StringVar(value="")

info_data = [
    ("Object Name:", obj_name_var, 0),
    ("Centroid X:", cx_var, 1),
    ("Centroid Y:", cy_var, 2),
    ("Angle (deg):", angle_var, 3),
    ("Detection Status:", status_var, 4),
]

for label_text, var, row in info_data:
    tk.Label(
        detection_frame,
        text=label_text,
        font=("Arial", 10, "bold"),
        anchor="w"
    ).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))

    tk.Label(
        detection_frame,
        textvariable=var,
        font=("Arial", 10),
        anchor="w",
        fg="#333333"
    ).grid(row=row, column=1, sticky="w", pady=4)

load_latest_detection()
root.mainloop()
