# Watchcase AI

Watchcase AI detects watch cases from a live camera feed and produces the position and orientation data needed for downstream robotic handling.

The application combines a custom YOLO segmentation model with OpenCV. It supports multiple detected cases, reference-image management, centroid and angle calculation, a Tkinter control panel, and TCP communication for trigger-based detection.

## Features

- Live camera preview with detection overlays
- YOLO segmentation using `model/best.pt`
- OpenCV contour fallback when segmentation does not return a result
- Centroid coordinates and orientation angle for each detected case
- ORB-based reference image management in `references/`
- Trigger server for PC-to-PC integration on TCP port `5000`
- Detection results written to `latest_detection.json`
- Works from the source tree and supports frozen/PyInstaller path handling

## Project Layout

```text
model/best.pt             Trained YOLO model
references/               Watch-case reference images
src/angle.py              Centroid and orientation calculation
src/reference_manager.py Reference image loading and matching
src/socket_server.py      TCP trigger server
src/webcam_infer.py       Detection and result generation
ui/gui.py                 Tkinter desktop interface
latest_detection.json     Latest detection result
requirements.txt          Python dependencies
```

## Requirements

- Windows with Python 3.9 or newer
- A working webcam
- Python packages listed in `requirements.txt`
- The trained model at `model/best.pt`

Install the dependencies from PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run The Application

From the repository root:

```powershell
python ui/gui.py
```

The GUI loads the model, opens the default camera, and starts the TCP server. Use the reference controls in the interface to add or activate a reference image before detection.

## Trigger Protocol

The server binds to `0.0.0.0:5000`. A client can connect and send the text `TRIGGER`:

```python
import socket

with socket.create_connection(("<watchcase-pc-ip>", 5000), timeout=10) as client:
	client.sendall(b"TRIGGER")
	response = client.recv(65535).decode("utf-8")
	print(response)
```

The server returns the detection response as JSON. If no detection is available, it returns `NO_DETECTION`.

## Detection Output

`latest_detection.json` contains the latest result. A successful detection includes fields such as:

```json
{
	"detection_status": "Detected",
	"detections": [
		{
			"object_name": "case1",
			"centroid_x": 320,
			"centroid_y": 240,
			"angle": 12.5,
			"confidence": 0.91,
			"poly": [[100, 120], [140, 120], [140, 180]],
			"color_type": "yolo"
		}
	]
}
```

Coordinates are pixel positions in the camera frame, and `angle` is reported in degrees by the orientation calculation.

## Troubleshooting

- **Model not found:** confirm that `model/best.pt` exists relative to the project root.
- **Camera unavailable:** close other camera applications and verify the default camera index in `ui/gui.py`.
- **Port already in use:** stop the other process using TCP port `5000` before starting the GUI.
- **No reference match:** add a clear reference image and activate it from the GUI.
- **Windows firewall:** allow Python to accept inbound connections if the trigger client runs on another PC.

## Notes

Generated files such as Python caches and debug overlays are local runtime artifacts and are not required to run the application.
