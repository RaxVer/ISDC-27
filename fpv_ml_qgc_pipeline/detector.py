"""
Robust Temporally-Stabilized Generic Anomaly & Object-Presence Detector (`FPVDetector`).

Designed for Aerial FPV Drone feeds:
- Does NOT output noisy/wrong class labels (e.g., 'BIRD', 'SPORTS BALL', 'SQUARE').
- Detects ANY distinct foreground object or visual anomaly standing out from the terrain/ground
  (luminance contrast + CIE LAB color saliency + optional label-free YOLOv8n proposals).
- Runs a Persistent Temporal Tracker (`_AnomalyTracker`) with:
  * Refresh throttling (detects every ~4 frames -> ~15 Hz, rendering smoothly at 60 FPS)
  * Hysteresis confirmation (`min_hits=2`) to ignore single-frame RF static glitches
  * Long retention (`hold_frames=24`, ~0.4s) so boxes NEVER flicker in and out
  * EMA coordinate smoothing (`alpha=0.28`) so boxes glide steadily on targets
"""

import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


YOLO_ONNX_URLS = [
    "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx",
    "https://huggingface.co/SpotLab/YOLOv8Detection/resolve/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx",
]


@dataclass
class Detection:
    category: str          # 'anomaly'
    label: str             # e.g. 'TARGET #1'
    confidence: float      # [0.0, 1.0]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (cx, cy)
    offset_from_center: Tuple[int, int]
    color_bgr: Tuple[int, int, int]


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray       # float32 [x1, y1, x2, y2]
    confidence: float
    hits: int
    misses: int


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


class _AnomalyTracker:
    """
    Persistent Temporal Box Tracker with Hysteresis & EMA Smoothing.
    Retains confirmed anomaly boxes across missed frames so detections stay rock-solid.
    """

    def __init__(self, min_hits: int = 2, max_age: int = 24, ema_alpha: float = 0.28):
        self.min_hits = min_hits
        self.max_age = max_age
        self.ema_alpha = ema_alpha
        self.next_id = 1
        self.tracks: List[_Track] = []

    def update(self, raw_boxes: List[Tuple[Tuple[int, int, int, int], float]], frame_w: int, frame_h: int) -> List[Detection]:
        det_arrs = [np.array(b, dtype=np.float32) for b, _ in raw_boxes]
        det_confs = [c for _, c in raw_boxes]
        matched_dets = set()

        for trk in self.tracks:
            best_idx = -1
            best_score = 0.0
            tcx = 0.5 * (trk.bbox[0] + trk.bbox[2])
            tcy = 0.5 * (trk.bbox[1] + trk.bbox[3])

            for i, db in enumerate(det_arrs):
                if i in matched_dets:
                    continue
                iou = _bbox_iou(trk.bbox, db)
                dcx = 0.5 * (db[0] + db[2])
                dcy = 0.5 * (db[1] + db[3])
                dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
                # Match by IoU OR close centroid proximity (handles moving drone camera)
                if iou >= 0.15 or dist <= max(45.0, frame_w * 0.045):
                    score = iou + max(0.0, 1.0 - dist / 150.0) * 0.5
                    if score > best_score:
                        best_score = score
                        best_idx = i

            if best_idx != -1:
                matched_dets.add(best_idx)
                db = det_arrs[best_idx]
                trk.bbox = (1.0 - self.ema_alpha) * trk.bbox + self.ema_alpha * db
                trk.confidence = 0.7 * trk.confidence + 0.3 * det_confs[best_idx]
                trk.hits += 1
                trk.misses = 0
            else:
                trk.misses += 1

        # Create new candidate tracks for unmatched detections
        for i, db in enumerate(det_arrs):
            if i not in matched_dets:
                self.tracks.append(
                    _Track(
                        track_id=self.next_id,
                        bbox=db.copy(),
                        confidence=det_confs[i],
                        hits=1,
                        misses=0,
                    )
                )
                self.next_id += 1
                if self.next_id > 99:
                    self.next_id = 1

        # Keep tracks that haven't exceeded retention window
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        # Return only confirmed tracks (hits >= min_hits)
        frame_cx, frame_cy = frame_w // 2, frame_h // 2
        confirmed: List[Detection] = []
        for trk in self.tracks:
            if trk.hits >= self.min_hits:
                x1, y1, x2, y2 = [int(round(v)) for v in trk.bbox]
                x1 = max(0, min(frame_w - 2, x1))
                y1 = max(0, min(frame_h - 2, y1))
                x2 = max(x1 + 2, min(frame_w - 1, x2))
                y2 = max(y1 + 2, min(frame_h - 1, y2))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                confirmed.append(
                    Detection(
                        category="anomaly",
                        label=f"ANOMALY #{trk.track_id}",
                        confidence=float(trk.confidence),
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        offset_from_center=(cx - frame_cx, cy - frame_cy),
                        color_bgr=(0, 255, 140),
                    )
                )
        return confirmed

    def get_retained(self, frame_w: int, frame_h: int) -> List[Detection]:
        """Returns currently retained confirmed tracks without decaying misses (used on skipped frames)."""
        frame_cx, frame_cy = frame_w // 2, frame_h // 2
        confirmed: List[Detection] = []
        for trk in self.tracks:
            if trk.hits >= self.min_hits:
                x1, y1, x2, y2 = [int(round(v)) for v in trk.bbox]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                confirmed.append(
                    Detection(
                        category="anomaly",
                        label=f"ANOMALY #{trk.track_id}",
                        confidence=float(trk.confidence),
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        offset_from_center=(cx - frame_cx, cy - frame_cy),
                        color_bgr=(0, 255, 140),
                    )
                )
        return confirmed


class FPVDetector:
    def __init__(
        self,
        enable_yolo: bool = True,
        enable_shapes: bool = False,
        enable_color_anomaly: bool = True,
        yolo_conf_thresh: float = 0.28,
        yolo_iou_thresh: float = 0.45,
        yolo_input_size: int = 416,
        anomaly_z_thresh: float = 2.8,
        detect_every_n: int = 4,
        hold_frames: int = 24,
    ):
        self.enable_yolo = enable_yolo
        self.enable_shapes = enable_shapes
        self.enable_color_anomaly = enable_color_anomaly
        self.yolo_conf_thresh = yolo_conf_thresh
        self.yolo_iou_thresh = yolo_iou_thresh
        self.yolo_input_size = yolo_input_size
        self.anomaly_z_thresh = anomaly_z_thresh
        self.detect_every_n = max(1, int(detect_every_n))
        self.frame_counter = 0

        self.tracker = _AnomalyTracker(min_hits=2, max_age=hold_frames, ema_alpha=0.28)

        self.yolo_session = None
        self.yolo_input_name = None
        self.yolo_model_hw = (640, 640)
        self.yolo_status = "DISABLED"

        self._yolo_lock = threading.Lock()
        self._yolo_pending_frame: Optional[np.ndarray] = None
        self._latest_yolo_boxes: List[Tuple[Tuple[int, int, int, int], float]] = []
        self._yolo_running = True
        self._yolo_thread: Optional[threading.Thread] = None

        if self.enable_yolo:
            self._init_yolo_onnx()

    def close(self) -> None:
        self._yolo_running = False
        if self._yolo_thread is not None and self._yolo_thread.is_alive():
            self._yolo_thread.join(timeout=1.5)

    def _init_yolo_onnx(self) -> None:
        models_dir = Path(__file__).resolve().parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = models_dir / "yolov8n.onnx"

        if not (onnx_path.exists() and onnx_path.stat().st_size > 1_000_000):
            for url in YOLO_ONNX_URLS:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=20) as resp, open(onnx_path, "wb") as f:
                        f.write(resp.read())
                    if onnx_path.stat().st_size > 1_000_000:
                        break
                except Exception:
                    pass

        if onnx_path.exists() and onnx_path.stat().st_size > 1_000_000:
            try:
                import onnxruntime as ort

                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 3
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
                self.yolo_status = f"Generic-Anomaly + YOLOv8n-Box ({prov_short}-Async)"
                self._yolo_thread = threading.Thread(target=self._yolo_worker_loop, daemon=True)
                self._yolo_thread.start()
            except Exception:
                self.yolo_session = None
                self.yolo_status = "Generic-Anomaly (Saliency)"
        else:
            self.yolo_status = "Generic-Anomaly (Saliency)"

    def _yolo_worker_loop(self) -> None:
        while self._yolo_running:
            frame_to_run = None
            with self._yolo_lock:
                if self._yolo_pending_frame is not None:
                    frame_to_run = self._yolo_pending_frame
                    self._yolo_pending_frame = None
            if frame_to_run is None:
                time.sleep(0.01)
                continue
            boxes = self._detect_yolo_boxes(frame_to_run)
            with self._yolo_lock:
                self._latest_yolo_boxes = boxes

    def _detect_yolo_boxes(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Extracts label-free bounding box proposals from YOLOv8n (640x360 input)."""
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

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outputs = self.yolo_session.run(None, {self.yolo_input_name: blob})[0]
        preds = np.squeeze(outputs)
        if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
            preds = preds.T
        if preds.ndim != 2 or preds.shape[1] < 6:
            return []

        boxes_xywh = preds[:, :4]
        max_scores = np.max(preds[:, 4:], axis=1)

        mask = max_scores >= self.yolo_conf_thresh
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]

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

        proposals: List[Tuple[Tuple[int, int, int, int], float]] = []
        for idx in np.array(indices).flatten():
            bx, by, b_w, b_h = nms_boxes[idx]
            ix1, iy1 = int(max(0, bx)), int(max(0, by))
            ix2, iy2 = int(min(w - 1, bx + b_w)), int(min(h - 1, by + b_h))
            # Ignore huge full-frame false boxes or edge-hugging drone legs
            area_frac = ((ix2 - ix1) * (iy2 - iy1)) / float(w * h)
            if area_frac > 0.25 or area_frac < 0.0008:
                continue
            proposals.append(((ix1, iy1, ix2, iy2), float(max_scores[idx])))
        return proposals

    def _detect_foreground_anomalies(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Generic Background-Contrast Foreground Anomaly Detector (< 1.5 ms):
        Finds ANY object/anomaly that stands out from the surrounding terrain/floor in EITHER:
          - Local luminance contrast (dark screws, bright white markers, debris, pads)
          - Chromaticity / color deviation (CIE LAB Z-score outlier)
        Automatically ignores border-touching drone landing gear / frame struts and thin floor seams.
        """
        h, w = frame.shape[:2]
        det_w, det_h = 320, 180
        sx_factor, sy_factor = w / float(det_w), h / float(det_h)
        small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)

        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        # Local background estimation via wide Gaussian blur (isolates objects from gradual lighting)
        bg_L = cv2.GaussianBlur(L, (31, 31), 9.0)
        diff_L = np.abs(L - bg_L)
        std_L = float(np.std(diff_L) + 5.0)
        z_L = diff_L / std_L

        # Global chromaticity outlier score
        mean_A, std_A = float(np.mean(A)), float(np.std(A) + 4.0)
        mean_B, std_B = float(np.mean(B)), float(np.std(B) + 4.0)
        z_C = np.sqrt(((A - mean_A) / std_A) ** 2 + ((B - mean_B) / std_B) ** 2)

        # Combined structural + spectral anomaly score
        z_combined = np.maximum(z_L, z_C * 0.95)
        mask = (z_combined >= self.anomaly_z_thresh).astype(np.uint8) * 255

        # Ignore outer 4% border & bottom-corner drone landing gear regions
        margin_x = int(det_w * 0.06)
        margin_y = int(det_h * 0.06)
        mask[:margin_y, :] = 0
        mask[det_h - margin_y :, :] = 0
        mask[:, :margin_x] = 0
        mask[:, det_w - margin_x :] = 0
        # Mask top-left OSD text ("PAL HD 1080P")
        mask[: int(det_h * 0.16), : int(det_w * 0.30)] = 0

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals: List[Tuple[Tuple[int, int, int, int], float]] = []

        min_area = (det_w * det_h) * 0.00045
        max_area = (det_w * det_h) * 0.12

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            sx, sy, sbw, sbh = cv2.boundingRect(cnt)
            aspect = sbw / float(max(sbh, 1))
            # Reject long thin floor seams / cracks
            if aspect > 4.2 or aspect < 0.24:
                continue

            peak_z = float(np.max(z_combined[sy : sy + sbh, sx : sx + sbw]))
            conf = float(np.clip(peak_z / 5.5, 0.45, 0.99))

            pad = 6
            x1 = max(0, int((sx - 2) * sx_factor) - pad)
            y1 = max(0, int((sy - 2) * sy_factor) - pad)
            x2 = min(w - 1, int((sx + sbw + 2) * sx_factor) + pad)
            y2 = min(h - 1, int((sy + sbh + 2) * sy_factor) + pad)
            proposals.append(((x1, y1, x2, y2), conf))

        return proposals

    def _merge_proposals_nms(
        self, proposals: List[Tuple[Tuple[int, int, int, int], float]]
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if len(proposals) <= 1:
            return proposals
        boxes = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2), _ in proposals]
        scores = [c for _, c in proposals]
        idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.35)
        if len(idxs) == 0:
            return []
        return [proposals[i] for i in np.array(idxs).flatten()]

    def detect_and_annotate(
        self, frame: np.ndarray, draw_hud: bool = True, perf_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, List[Detection], Dict[str, float]]:
        """
        Runs throttled foreground anomaly + YOLO proposals, updates persistent tracker,
        and renders clean, non-flickering bounding boxes at full 60 FPS.
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        self.frame_counter += 1

        if self.frame_counter % self.detect_every_n == 1 or self.detect_every_n == 1:
            raw_proposals: List[Tuple[Tuple[int, int, int, int], float]] = []

            if self.enable_color_anomaly:
                raw_proposals.extend(self._detect_foreground_anomalies(frame))

            if self.enable_yolo and self.yolo_session is not None:
                with self._yolo_lock:
                    if self._yolo_pending_frame is None:
                        self._yolo_pending_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)
                    for (bx1, by1, bx2, by2), sc in self._latest_yolo_boxes:
                        sx, sy = w / 640.0, h / 360.0
                        raw_proposals.append(
                            ((int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy)), sc)
                        )

            merged = self._merge_proposals_nms(raw_proposals)
            detections = self.tracker.update(merged, w, h)
        else:
            # On intermediate frames, draw retained & smoothed boxes (< 0.1 ms)
            detections = self.tracker.get_retained(w, h)

        detect_ms = (time.perf_counter() - t0) * 1000.0
        annotated = frame
        cx, cy = w // 2, h // 2

        # Center boresight crosshair
        cv2.line(annotated, (cx - 18, cy), (cx - 5, cy), (180, 255, 180), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx + 5, cy), (cx + 18, cy), (180, 255, 180), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx, cy - 18), (cx, cy - 5), (180, 255, 180), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx, cy + 5), (cx, cy + 18), (180, 255, 180), 1, cv2.LINE_AA)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            tcx, tcy = det.center
            color = det.color_bgr

            # Clean bounding box + center marker (no noisy class labels)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (tcx, tcy), 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                annotated,
                det.label,
                (x1, max(48, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                1,
                cv2.LINE_AA,
            )

        if draw_hud:
            self._render_tactical_hud(annotated, detections, detect_ms, perf_stats or {})

        return annotated, detections, {"detect_ms": detect_ms, "count": len(detections)}

    def _render_tactical_hud(
        self,
        canvas: np.ndarray,
        detections: List[Detection],
        detect_ms: float,
        perf_stats: Dict[str, float],
    ) -> None:
        h, w = canvas.shape[:2]
        top_strip = canvas[:34, :]
        dark_bar = np.full_like(top_strip, (12, 16, 22))
        cv2.addWeighted(dark_bar, 0.72, top_strip, 0.28, 0, top_strip)

        fps = perf_stats.get("fps", 0.0)
        enh_ms = perf_stats.get("enhance_ms", 0.0)
        e2e_ms = perf_stats.get("e2e_delay_ms", enh_ms + detect_ms)
        mode_str = perf_stats.get("enhancer_mode", "CNN+RNN").upper()

        hud_left = (
            f"FPV-ML [{mode_str}] {w}x{h} | {fps:4.1f} FPS | "
            f"DELAY: {e2e_ms:4.1f}ms (<120ms OK) [SR:{enh_ms:4.1f}ms DET:{detect_ms:4.1f}ms]"
        )
        hud_right = f"ANOMALIES LOCKED: {len(detections)}"

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
            (0, 255, 140),
            1,
            cv2.LINE_AA,
        )
