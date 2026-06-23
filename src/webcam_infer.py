import cv2
import numpy as np
import json
import os
import sys
from ultralytics import YOLO
from src.reference_manager import ReferenceManager
from src.angle import get_angle

def run_detection_loop(model, ref_mgr, detection_file):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    print("Press S to detect")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not detected")
            break

        display = frame.copy()
        cv2.putText(
            display,
            "Press S to Detect | Q to Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.imshow("Watch Case Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('s'):
            active_name = ref_mgr.get_active_name()
            result_img = frame.copy()
            results = model(frame, conf=0.25)

            yolo_found = False
            detected_name = None
            centroid_x = None
            centroid_y = None
            angle = 0.0

            for r in results:
                if r.masks is None:
                    continue
                for i, poly in enumerate(r.masks.xy):
                    yolo_found = True
                    poly = np.array(poly, dtype=np.int32)
                    cv2.polylines(result_img, [poly], True, (0, 255, 0), 2)

                    M = cv2.moments(poly)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    try:
                        print("Contour points:", len(poly))
                        angle = get_angle(poly)
                        print("Calculated angle:", angle)
                    except Exception as e:
                        print("Angle calculation error:", e)
                        angle = 0.0

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    x, y, w, h = cv2.boundingRect(poly)
                    object_roi = gray[y:y+h, x:x+w]
                    ref_mgr.match(object_roi)

                    final_name = active_name if active_name else "No Reference Set"
                    detected_name = final_name
                    centroid_x = cx
                    centroid_y = cy

                    cv2.circle(result_img, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(
                        result_img,
                        f"{final_name} ({cx},{cy}) A:{angle:.1f}",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    print(f"\nObject Name: {final_name}")
                    print(f"Centroid X: {cx}")
                    print(f"Centroid Y: {cy}")
                    print(f"Angle: {angle} deg")
                    print(f"Status: Detected")

            if not yolo_found:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest)
                    if area > 1000:
                        M = cv2.moments(largest)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            try:
                                print("Contour points:", len(largest))
                                angle = get_angle(largest)
                                print("Calculated angle:", angle)
                            except Exception as e:
                                print("Angle calculation error:", e)
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
                            detected_name = final_name
                            centroid_x = cx
                            centroid_y = cy

                            cv2.drawContours(result_img, [largest], -1, (255, 0, 255), 2)
                            cv2.circle(result_img, (cx, cy), 6, (0, 0, 255), -1)
                            cv2.putText(
                                result_img,
                                f"{final_name} ({cx},{cy}) A:{angle:.1f}",
                                (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 255),
                                2
                            )

                            print(f"Object Name: {final_name}")
                            print(f"Centroid X: {cx}")
                            print(f"Centroid Y: {cy}")
                            print(f"Angle: {angle} deg")
                            print(f"Status: Detected")
                else:
                    print("No object found")

            if detected_name is not None:
                data = {
                       "object_name": detected_name,
                       "centroid_x": centroid_x,
                       "centroid_y": centroid_y,
                       "angle": angle,
                       "detection_status": "Detected"
                        }
                with open(detection_file, "w") as f:
                    json.dump(data, f)
                cv2.imshow("Detection Result", result_img)
                print("\nPress S for next detection")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
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

    model = YOLO(resource_path("model/best.pt"))
    ref_mgr = ReferenceManager(ref_dir=data_path("references"))
    detection_file = data_path("latest_detection.json")
    run_detection_loop(model, ref_mgr, detection_file)
