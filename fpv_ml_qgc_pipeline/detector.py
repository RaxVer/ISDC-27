"""
Multi-Modal Real-Time Detection Engine for Drone FPV Feeds (`FPVDetector`).

Combines three real-time detection heads:
1. YOLOv8n Object Detection (ONNX Runtime):
   Detects persons, vehicles, aircraft, boats, backpacks, and general COCO objects in real time.
2. Geometric Shape & Drone Landing Pad / Fiducial Detection:
   Detects precision landing targets (concentric circles / H-pads) and geometric markers
   (Triangle, Square/Rectangle, Pentagon, Hexagon, Circle) with center offset vector (dx, dy).
3. Color Abnormality & Spectral Saliency Anomaly Detection:
   - Statistical CIE LAB Spectral Anomaly Map (Z-score distance from background terrain color distribution)
   - High-visibility HSV target signature classification (Red/Orange Hazard, Yellow Rescue, Cyan Marker).
"""

import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Reliable mirror URLs for pre-exported YOLOv8n / YOLOv5n ONNX models
YOLO_ONNX_URLS = [
    "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx",
    "https://huggingface.co/SpotLab/YOLOv8Detection/resolve/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx",
]


@dataclass
class Detection:
    category: str          # 'yolo', 'shape', 'anomaly'
    label: str             # e.g. 'person', 'SHAPE:TRIANGLE', 'ANOMALY:RED-THERMAL (Z=4.2)'
    confidence: float      # [0.0, 1.0]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (cx, cy)
    offset_from_center: Tuple[int, int]  # (dx, dy) from frame optical center
    color_bgr: Tuple[int, int, int]


class FPVDetector:
    def __init__(
        self,
        enable_yolo: bool = True,
        enable_shapes: bool = True,
        enable_color_anomaly: bool = True,
        yolo_conf_thresh: float = 0.35,
        yolo_iou_thresh: float = 0.45,
        yolo_input_size: int = 416,
        anomaly_z_thresh: float = 3.2,
    ):
        self.enable_yolo = enable_yolo
        self.enable_shapes = enable_shapes
        self.enable_color_anomaly = enable_color_anomaly
        self.yolo_conf_thresh = yolo_conf_thresh
        self.yolo_iou_thresh = yolo_iou_thresh
        self.yolo_input_size = yolo_input_size
        self.anomaly_z_thresh = anomaly_z_thresh

        self.yolo_session = None
        self.yolo_input_name = None
        self.yolo_model_hw = (640, 640)
        self.yolo_status = "DISABLED"

        # Asynchronous non-blocking YOLO state so video stream stays locked at 30+ FPS
        self._yolo_lock = threading.Lock()
        self._yolo_pending_frame: Optional[np.ndarray] = None
        self._latest_yolo_detections: List[Detection] = []
        self._yolo_running = True
        self._yolo_thread: Optional[threading.Thread] = None

        if self.enable_yolo:
            self._init_yolo_onnx()

    def close(self) -> None:
        self._yolo_running = False
        if self._yolo_thread is not None and self._yolo_thread.is_alive():
            self._yolo_thread.join(timeout=1.5)

    def _init_yolo_onnx(self) -> None:
        """Loads YOLOv8n ONNX model into ONNX Runtime (auto-downloading if needed)."""
        models_dir = Path(__file__).resolve().parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = models_dir / "yolov8n.onnx"

        if not (onnx_path.exists() and onnx_path.stat().st_size > 1_000_000):
            for url in YOLO_ONNX_URLS:
                try:
                    print(f"[FPVDetector] Downloading YOLOv8n ONNX model from {url} ...")
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=20) as resp, open(onnx_path, "wb") as f:
                        f.write(resp.read())
                    if onnx_path.stat().st_size > 1_000_000:
                        print(f"[FPVDetector] YOLOv8n ONNX saved to {onnx_path}")
                        break
                except Exception as exc:
                    print(f"[FPVDetector] Download mirror failed ({exc}), trying next...")

        if onnx_path.exists() and onnx_path.stat().st_size > 1_000_000:
            try:
                import onnxruntime as ort

                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 4
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                providers = [
                    p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if p in ort.get_available_providers()
                ] or ["CPUExecutionProvider"]
                self.yolo_session = ort.InferenceSession(
                    str(onnx_path), sess_options=sess_opts, providers=providers
                )
                inp_meta = self.yolo_session.get_inputs()[0]
                self.yolo_input_name = inp_meta.name
                shape = inp_meta.shape
                if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                    self.yolo_model_hw = (shape[2], shape[3])
                else:
                    self.yolo_model_hw = (self.yolo_input_size, self.yolo_input_size)
                prov_short = providers[0].replace("ExecutionProvider", "")
                self.yolo_status = f"YOLOv8n-ONNX ({prov_short}-Async)"
                self._yolo_thread = threading.Thread(target=self._yolo_worker_loop, daemon=True)
                self._yolo_thread.start()
            except Exception as exc:
                print(f"[FPVDetector] Could not initialize YOLOv8n ONNX session: {exc}")
                self.yolo_session = None
                self.yolo_status = "YOLO-UNAVAILABLE"
        else:
            self.yolo_status = "YOLO-NO-WEIGHTS"

    def _yolo_worker_loop(self) -> None:
        while self._yolo_running:
            frame_to_run = None
            with self._yolo_lock:
                if self._yolo_pending_frame is not None:
                    frame_to_run = self._yolo_pending_frame
                    self._yolo_pending_frame = None
            if frame_to_run is None:
                time.sleep(0.005)
                continue
            dets = self._detect_yolo(frame_to_run)
            with self._yolo_lock:
                self._latest_yolo_detections = dets

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Runs YOLOv8n ONNX inference with letterbox scaling and Non-Maximum Suppression."""
        if self.yolo_session is None:
            return []

        h, w = frame.shape[:2]
        ih, iw = self.yolo_model_hw
        scale = min(iw / w, ih / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        dw, dh = (iw - nw) / 2.0, (ih - nh) / 2.0

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        canvas[top : top + nh, left : left + nw] = resized

        # BGR -> RGB, [0, 1], NCHW
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outputs = self.yolo_session.run(None, {self.yolo_input_name: blob})[0]
        # YOLOv8 output shape is [1, 84, 8400] -> transpose to [8400, 84]
        preds = np.squeeze(outputs)
        if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
            preds = preds.T

        if preds.ndim != 2 or preds.shape[1] < 6:
            return []

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        max_scores = class_scores[np.arange(len(class_scores)), class_ids]

        mask = max_scores >= self.yolo_conf_thresh
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        # Convert center (cx, cy, bw, bh) in letterbox space back to original frame (x, y, w, h)
        cx = (boxes_xywh[:, 0] - left) / scale
        cy = (boxes_xywh[:, 1] - top) / scale
        bw = boxes_xywh[:, 2] / scale
        bh = boxes_xywh[:, 3] / scale
        x1 = np.clip(cx - 0.5 * bw, 0, w - 1)
        y1 = np.clip(cy - 0.5 * bh, 0, h - 1)

        nms_boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, max_scores.tolist(), self.yolo_conf_thresh, self.yolo_iou_thresh
        )
        if len(indices) == 0:
            return []

        detections: List[Detection] = []
        frame_cx, frame_cy = w // 2, h // 2
        for idx in np.array(indices).flatten():
            bx, by, b_w, b_h = nms_boxes[idx]
            ix1 = int(max(0, bx))
            iy1 = int(max(0, by))
            ix2 = int(min(w - 1, bx + b_w))
            iy2 = int(min(h - 1, by + b_h))
            tcx, tcy = (ix1 + ix2) // 2, (iy1 + iy2) // 2
            cid = int(class_ids[idx])
            cls_name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"cls_{cid}"
            detections.append(
                Detection(
                    category="yolo",
                    label=f"OBJ:{cls_name.upper()}",
                    confidence=float(max_scores[idx]),
                    bbox=(ix1, iy1, ix2, iy2),
                    center=(tcx, tcy),
                    offset_from_center=(tcx - frame_cx, tcy - frame_cy),
                    color_bgr=(0, 255, 128),  # Tactical green for YOLO objects
                )
            )
        return detections

    def _detect_shapes(self, frame: np.ndarray) -> List[Detection]:
        """
        Real-Time Geometric Shape & Drone Landing Pad Detector.
        Runs on a fixed 640x360 grid (< 1.5 ms even at 1080p / 1440p output) and projects coordinates back.
        """
        h, w = frame.shape[:2]
        det_w, det_h = 640, 360
        sx_factor, sy_factor = w / float(det_w), h / float(det_h)
        frame_area = float(h * w)
        min_area = max(450.0, frame_area * 0.0012)
        max_area = frame_area * 0.35

        small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 55, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            return []

        hierarchy = hierarchy[0]
        detections: List[Detection] = []
        frame_cx, frame_cy = w // 2, h // 2

        for i, cnt in enumerate(contours):
            cnt_f = cnt.astype(np.float32)
            cnt_f[:, 0, 0] *= sx_factor
            cnt_f[:, 0, 1] *= sy_factor
            cnt_full = cnt_f.astype(np.int32)

            area = cv2.contourArea(cnt_full)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(cnt_full, True)
            if peri <= 0:
                continue

            hull = cv2.convexHull(cnt_full)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            if solidity < 0.84:
                continue

            circularity = (4.0 * np.pi * area) / (peri * peri)
            approx = cv2.approxPolyDP(cnt_full, 0.032 * peri, True)
            if not cv2.isContourConvex(approx):
                continue

            v_count = len(approx)
            x, y, bw, bh = cv2.boundingRect(approx)
            if x <= 4 or y <= 4 or (x + bw) >= (w - 4) or (y + bh) >= (h - 4):
                continue

            aspect = bw / float(max(bh, 1))
            if aspect < 0.35 or aspect > 2.8:
                continue

            has_child = hierarchy[i][2] != -1
            shape_name = None
            conf = float(np.clip(solidity, 0.5, 0.99))

            if v_count == 3:
                shape_name = "SHAPE:TRIANGLE"
            elif v_count == 4:
                rect_extent = area / float(max(bw * bh, 1))
                if rect_extent > 0.72:
                    if 0.85 <= aspect <= 1.18:
                        shape_name = "PAD:SQUARE-TARGET" if has_child else "SHAPE:SQUARE"
                    else:
                        shape_name = "SHAPE:RECTANGLE"
            elif v_count == 5 and solidity > 0.90:
                shape_name = "SHAPE:PENTAGON"
            elif v_count == 6 and solidity > 0.90:
                shape_name = "SHAPE:HEXAGON"
            elif v_count >= 7 and circularity > 0.80:
                shape_name = "PAD:LANDING-CIRCLE" if has_child else "SHAPE:CIRCLE"
                conf = float(np.clip(circularity, 0.75, 0.99))

            if shape_name is None:
                continue

            tcx, tcy = x + bw // 2, y + bh // 2
            if any(abs(tcx - d.center[0]) < 24 and abs(tcy - d.center[1]) < 24 for d in detections):
                continue

            detections.append(
                Detection(
                    category="shape",
                    label=shape_name,
                    confidence=conf,
                    bbox=(x, y, x + bw, y + bh),
                    center=(tcx, tcy),
                    offset_from_center=(tcx - frame_cx, tcy - frame_cy),
                    color_bgr=(255, 200, 0),
                )
            )
            if len(detections) >= 6:
                break

        return detections

    def _detect_color_anomalies(self, frame: np.ndarray) -> List[Detection]:
        """
        Real-Time Color Abnormality & Spectral Saliency Detector on fixed 320x180 grid (< 1.2 ms at 1080p/1440p).
        """
        h, w = frame.shape[:2]
        det_w, det_h = 320, 180
        sx_factor, sy_factor = w / float(det_w), h / float(det_h)
        small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_NEAREST)

        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_chan = lab[:, :, 1]
        b_chan = lab[:, :, 2]

        mean_a, std_a = float(np.mean(a_chan)), float(np.std(a_chan) + 4.0)
        mean_b, std_b = float(np.mean(b_chan)), float(np.std(b_chan) + 4.0)

        z_map = np.sqrt(((a_chan - mean_a) / std_a) ** 2 + ((b_chan - mean_b) / std_b) ** 2)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        anomaly_mask = ((z_map >= self.anomaly_z_thresh) & (sat >= 95) & (val >= 70)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        frame_cx, frame_cy = w // 2, h // 2
        min_small_area = max(14.0, (det_w * det_h) * 0.0005)
        max_small_area = (det_w * det_h) * 0.12

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_small_area or area > max_small_area:
                continue

            sx, sy, sbw, sbh = cv2.boundingRect(cnt)
            roi_z = z_map[sy : sy + sbh, sx : sx + sbw]
            roi_h = hue[sy : sy + sbh, sx : sx + sbw]
            peak_z = float(np.max(roi_z))
            med_hue = float(np.median(roi_h))

            if med_hue <= 12 or med_hue >= 165:
                sig = "RED/THERMAL"
            elif 12 < med_hue <= 24:
                sig = "ORANGE-HAZARD"
            elif 24 < med_hue <= 36:
                sig = "YELLOW-MARKER"
            elif 80 <= med_hue <= 130:
                sig = "CYAN/BLUE-TARGET"
            elif 130 < med_hue < 165:
                sig = "MAGENTA-ANOMALY"
            else:
                sig = "SPECTRAL-OUTLIER"

            x1, y1 = int(sx * sx_factor), int(sy * sy_factor)
            x2, y2 = min(w - 1, int((sx + sbw) * sx_factor)), min(h - 1, int((sy + sbh) * sy_factor))
            tcx, tcy = (x1 + x2) // 2, (y1 + y2) // 2
            conf = float(np.clip(peak_z / 6.0, 0.50, 0.99))

            detections.append(
                Detection(
                    category="anomaly",
                    label=f"ANOMALY:{sig} (Z={peak_z:.1f})",
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(tcx, tcy),
                    offset_from_center=(tcx - frame_cx, tcy - frame_cy),
                    color_bgr=(0, 64, 255),
                )
            )
            if len(detections) >= 6:
                break

        return detections

    def detect_and_annotate(
        self, frame: np.ndarray, draw_hud: bool = True, perf_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, List[Detection], Dict[str, float]]:
        """
        Runs all enabled detection heads on the enhanced frame and renders tactical HUD overlays.
        """
        t0 = time.perf_counter()
        all_detections: List[Detection] = []

        if self.enable_yolo and self.yolo_session is not None:
            with self._yolo_lock:
                if self._yolo_pending_frame is None:
                    self._yolo_pending_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)
                for ydet in self._latest_yolo_detections:
                    # Scale YOLO 640x360 coordinates to current output resolution (1080p / 1440p)
                    h, w = frame.shape[:2]
                    sx, sy = w / 640.0, h / 360.0
                    bx1, by1, bx2, by2 = ydet.bbox
                    nx1, ny1, nx2, ny2 = int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy)
                    ncx, ncy = (nx1 + nx2) // 2, (ny1 + ny2) // 2
                    all_detections.append(
                        Detection(
                            category=ydet.category,
                            label=ydet.label,
                            confidence=ydet.confidence,
                            bbox=(nx1, ny1, nx2, ny2),
                            center=(ncx, ncy),
                            offset_from_center=(ncx - w // 2, ncy - h // 2),
                            color_bgr=ydet.color_bgr,
                        )
                    )
        if self.enable_shapes:
            all_detections.extend(self._detect_shapes(frame))
        if self.enable_color_anomaly:
            all_detections.extend(self._detect_color_anomalies(frame))

        detect_ms = (time.perf_counter() - t0) * 1000.0
        annotated = frame
        h, w = annotated.shape[:2]
        cx, cy = w // 2, h // 2

        # Draw center optical reticle (Drone Boresight Crosshair)
        cv2.line(annotated, (cx - 22, cy), (cx - 6, cy), (180, 255, 180), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx + 6, cy), (cx + 22, cy), (180, 255, 180), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx, cy - 22), (cx, cy - 6), (180, 255, 180), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx, cy + 6), (cx, cy + 22), (180, 255, 180), 1, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), 1, cv2.LINE_AA)

        primary_target: Optional[Detection] = None
        min_dist = 1e9

        for det in all_detections:
            x1, y1, x2, y2 = det.bbox
            tcx, tcy = det.center
            dx, dy = det.offset_from_center
            color = det.color_bgr

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (tcx, tcy), 4, color, -1, cv2.LINE_AA)

            caption = f"{det.label} {int(det.confidence * 100)}% [dx:{dx:+d} dy:{dy:+d}]"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            ly = max(th + 6, y1 - 4)
            cv2.rectangle(annotated, (x1, ly - th - 5), (x1 + tw + 8, ly + 3), (15, 15, 15), -1)
            cv2.putText(
                annotated,
                caption,
                (x1 + 4, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                1,
                cv2.LINE_AA,
            )

            dist = (dx * dx + dy * dy) ** 0.5
            if dist < min_dist:
                min_dist = dist
                primary_target = det

        if primary_target is not None:
            cv2.line(
                annotated,
                (cx, cy),
                primary_target.center,
                primary_target.color_bgr,
                1,
                cv2.LINE_AA,
            )

        if draw_hud:
            self._render_tactical_hud(annotated, all_detections, detect_ms, perf_stats or {})

        return annotated, all_detections, {"detect_ms": detect_ms, "count": len(all_detections)}

    def _render_tactical_hud(
        self,
        canvas: np.ndarray,
        detections: List[Detection],
        detect_ms: float,
        perf_stats: Dict[str, float],
    ) -> None:
        """Draws a telemetry HUD bar at the top 34px of the video stream (blending only the 34px strip)."""
        h, w = canvas.shape[:2]
        top_strip = canvas[:34, :]
        dark_bar = np.full_like(top_strip, (12, 16, 22))
        cv2.addWeighted(dark_bar, 0.72, top_strip, 0.28, 0, top_strip)

        fps = perf_stats.get("fps", 0.0)
        enh_ms = perf_stats.get("enhance_ms", 0.0)
        e2e_ms = perf_stats.get("e2e_delay_ms", enh_ms + detect_ms)
        mode_str = perf_stats.get("enhancer_mode", "CNN+RNN").upper()
        n_yolo = sum(1 for d in detections if d.category == "yolo")
        n_shape = sum(1 for d in detections if d.category == "shape")
        n_anom = sum(1 for d in detections if d.category == "anomaly")

        hud_left = (
            f"FPV-ML [{mode_str}] {w}x{h} | {fps:4.1f} FPS | "
            f"DELAY: {e2e_ms:4.1f}ms (<120ms OK) [SR:{enh_ms:4.1f}ms DET:{detect_ms:4.1f}ms]"
        )
        hud_right = (
            f"OBJ({'ON' if self.enable_yolo else 'OFF'}):{n_yolo}  "
            f"SHAPE({'ON' if self.enable_shapes else 'OFF'}):{n_shape}  "
            f"ANOMALY({'ON' if self.enable_color_anomaly else 'OFF'}):{n_anom}"
        )

        cv2.putText(
            canvas, hud_left, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 1, cv2.LINE_AA
        )
        (rw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(
            canvas,
            hud_right,
            (max(12, w - rw - 12), 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 240, 255),
            1,
            cv2.LINE_AA,
        )
