import cv2
import numpy as np
import json
import os
import sys
from ultralytics import YOLO
from src.reference_manager import ReferenceManager
from src.angle import get_angle

# Single Path Manager
file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)

def data_path(relative_path):
    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = project_root
    return os.path.join(base_path, relative_path)

def write_error_status(detection_file, message):
    try:
        with open(detection_file, "w") as f:
            json.dump({
                "detection_status": str(message),
                "detections": []
            }, f)
    except Exception:
        pass

def detect_on_frame(model, ref_mgr, frame, detection_file):
    """
    Run detection on a single frame.
    Called from a worker thread (no GUI operations allowed).
    Returns (True, annotated_BGR_image) if object found,
            (False, frame_with_debug_overlays) otherwise.
    """
    active_name = ref_mgr.get_active_name()
    result_img = frame.copy()
    results = model(frame, conf=0.25)

    yolo_found = False
    detections = []

    for r in results:
        if r.masks is None:
            continue
        for i, poly in enumerate(r.masks.xy):
            yolo_found = True
            poly = np.array(poly, dtype=np.int32)

            try:
                cx_pca, cy_pca, angle = get_angle(poly)
                cx = int(cx_pca)
                cy = int(cy_pca)
            except Exception:
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                angle = 0.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = cv2.boundingRect(poly)
            object_roi = gray[y:y+h, x:x+w]

            ref_mgr.match(object_roi)

            final_name = active_name if active_name else "No Reference Set"

            conf = 1.0
            if r.boxes is not None and r.boxes.conf is not None:
                if i < len(r.boxes.conf):
                    conf = float(r.boxes.conf[i].item())

            detections.append({
                "object_name": final_name,
                "centroid_x": cx,
                "centroid_y": cy,
                "angle": float(angle),
                "confidence": conf,
                "poly": poly,
                "color_type": "yolo"
            })

    if not yolo_found:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for largest in contours:
            area = cv2.contourArea(largest)
            if area > 1000:
                try:
                    cx_pca, cy_pca, angle = get_angle(largest)
                    cx = int(cx_pca)
                    cy = int(cy_pca)
                except Exception:
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    angle = 0.0

                x, y, w, h = cv2.boundingRect(largest)
                pad = 10
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(frame.shape[1] - x, w + 2 * pad)
                h = min(frame.shape[0] - y, h + 2 * pad)
                object_roi = gray[y:y + h, x:x + w]

                ref_mgr.match(object_roi)

                final_name = active_name if active_name else "No Reference Set"

                detections.append({
                    "object_name": final_name,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "angle": float(angle),
                    "confidence": 1.0,
                    "poly": largest,
                    "color_type": "classical"
                })

    # Sort detections from left to right using centroid_x
    detections.sort(key=lambda d: d["centroid_x"])

    # Keep only the first two detections
    detections = detections[:2]

    # Draw overlays for only these two detections
    for d in detections:
        poly = d["poly"]
        cx = d["centroid_x"]
        cy = d["centroid_y"]
        angle = d["angle"]
        final_name = d["object_name"]
        color_type = d["color_type"]

        if color_type == "yolo":
            print("[LOG] Entering draw section")
            cv2.polylines(result_img, [poly], True, (0, 255, 0), 2)
            print("[LOG] Contour drawn")
            cv2.circle(result_img, (cx, cy), 6, (0, 0, 255), -1)
            print("[LOG] Centroid drawn")
            cv2.putText(
                result_img,
                f"{final_name} ({cx},{cy}) A:{angle:.1f}",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            print("[LOG] Angle drawn")
        else:
            print("[LOG] Entering draw section")
            cv2.drawContours(result_img, [poly], -1, (255, 0, 255), 2)
            print("[LOG] Contour drawn")
            cv2.circle(result_img, (cx, cy), 6, (0, 0, 255), -1)
            print("[LOG] Centroid drawn")
            cv2.putText(
                result_img,
                f"{final_name} ({cx},{cy}) A:{angle:.1f}",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )
            print("[LOG] Angle drawn")

    if detections:
        # Save a debug image debug_overlay.jpg in the same directory as detection_file BEFORE json.dump
        debug_image_path = os.path.join(os.path.dirname(detection_file), "debug_overlay.jpg")
        cv2.imwrite(debug_image_path, result_img)
        print(f"[LOG] Display image updated: saved to {debug_image_path}")

        json_detections = []
        for d in detections:
            json_detections.append({
                "object_name": str(d["object_name"]),
                "centroid_x": int(d["centroid_x"]),
                "centroid_y": int(d["centroid_y"]),
                "angle": float(d["angle"])
            })

        data = {
            "detection_status": "Detected",
            "detections": json_detections
        }
        with open(detection_file, "w") as f:
            json.dump(data, f)
        return True, result_img

    return False, result_img


def run_detection_loop(model, ref_mgr, detection_file):
    """
    Standalone detection loop for __main__ execution only.
    Uses OpenCV HighGUI on the main thread (works correctly here).
    Not used when launched from gui.py.
    """
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        show_overlay = False
        overlay_image = None

        while True:
            ret, frame = cap.read()
            if not ret:
                write_error_status(detection_file, "Camera not found - check webcam connection")
                break

            if show_overlay and overlay_image is not None:
                cv2.imshow("Watch Case Detection", overlay_image)
            else:
                display = frame.copy()
                cv2.putText(
                    display,
                    "S: Detect | Q: Quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                cv2.imshow("Watch Case Detection", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break

            try:
                if cv2.getWindowProperty("Watch Case Detection", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key == ord('s'):
                # Briefly show the live feed so the user knows detection is running
                display = frame.copy()
                cv2.putText(display, "S: Detect | Q: Quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Watch Case Detection", display)
                cv2.waitKey(1)

                success, result_img = detect_on_frame(model, ref_mgr, frame, detection_file)
                if success:
                    show_overlay = True
                    overlay_image = result_img
                    overlay_path = os.path.join(os.path.dirname(detection_file), "last_detection_overlay.jpg")
                    cv2.imwrite(overlay_path, overlay_image)
                    print(f"[DEBUG] Overlay image saved: {overlay_path}")
                    cv2.imshow("Watch Case Detection", overlay_image)
                    cv2.waitKey(1)
                    print("[DEBUG] Showing image with overlays")
                else:
                    show_overlay = False
                    overlay_image = None
                    write_error_status(detection_file, "No object detected")

    except Exception as e:
        write_error_status(detection_file, f"Detection error: {e}")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = data_path("model/best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    model = YOLO(model_path)
    ref_mgr = ReferenceManager(ref_dir=data_path("references"))
    detection_file = data_path("latest_detection.json")
    run_detection_loop(model, ref_mgr, detection_file)
