# ISDC 27 — Real-Time FPV Drone ML Video Enhancer, Ground-Plane Anomaly Detector & QGroundControl Hub

End-to-end real-time Computer Vision, Super-Resolution (`CNN + Temporal RNN`), Bilateral De-Graining + `1.65x` Vibrancy Filter, and **Martian-Terrain Foreign-Object & Tool/Sensor Anomaly Detection** (`Dual-Scale Signed Ground-Plane Model + YOLO11s opset=12`) running at **1080p @ 60 FPS** (`~15 ms` processing delay, well under the `120 ms` budget).

---

## 1. Repository Structure

| File | Description |
| :--- | :--- |
| [`pipeline.py`](pipeline.py) | Main real-time runner (`1280x720` capture $\rightarrow$ `1920x1080 @ 60fps` enhancement $\rightarrow$ anomaly detection $\rightarrow$ full-res local display & optional QGroundControl H.264 RTP stream). |
| [`enhancer.py`](enhancer.py) | **Motion-Gated Temporal RNN** + **15x15 PAL Rainbow Cross-Color Killer** + **1.65x Chroma Vibrancy LUT** + **S-Curve Cinema Contrast** + **Bilateral De-Graining & `10/255` Edge-Cored Super-Resolution**. |
| [`detector.py`](detector.py) | **Dual-Scale Signed Ground-Plane Anomaly Detector** + **Official Ultralytics `YOLO11s` (`opset=12`, Async)** + **Ground-Arena Polygon Mask** + **Fast-Decay Temporal Box Tracker**. |
| [`streamer.py`](streamer.py) | Zero-latency 6-thread H.264 RTP (`pt=96`) & MPEG-TS UDP streamer (`127.0.0.1:5600` or LAN IPs) for QGroundControl (`--stream-qgc`). |
| [`capture.py`](capture.py) | Threaded lock-free V4L2 USB capture card reader (`RC285H -> AV2HDMI -> USB`), **Dual-Camera Side-by-Side / PiP Multiplexer**, and **Analog 5.8 GHz FPV Simulator**. |
| [`mavlink_hub.py`](mavlink_hub.py) | Optional multi-laptop D-Link LAN MAVLink Telemetry Router for forwarding role-specific Pixhawk telemetry to static laptop IPs. |
| [`models_arch.py`](models_arch.py) & [`train_enhancer.py`](train_enhancer.py) | ESPCN Sub-Pixel CNN architecture (`opset=12` ONNX export) and train/test PSNR evaluation script. |

---

## 2. Quick Start (Run on Your Ubuntu Laptop)

```bash
cd ~/ISDC-27/fpv_ml_qgc_pipeline

# Run live on RC285H -> AV2HDMI -> USB Capture Card (/dev/video2) in full 1080p60:
./.venv/bin/python pipeline.py --source /dev/video2

# Run on built-in webcam (/dev/video0):
./.venv/bin/python pipeline.py --source /dev/video0

# Run with simulated 5.8 GHz FPV drone feed + split-screen comparison:
./.venv/bin/python pipeline.py --source sim --split-view

# Stream live to QGroundControl (UDP h.264 Video Stream, Port 5600):
./.venv/bin/python pipeline.py --source /dev/video2 --stream-qgc
```
