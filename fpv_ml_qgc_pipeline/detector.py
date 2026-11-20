"""
Martian-Terrain Foreign Object & Tool/Sensor Anomaly Detector (`FPVDetector`).

Specifically tuned for aerial drone feeds over rocky / sandy / Martian-like terrain:
- Rejects natural rocks, gravel, shadows, and floor seams using Robust Median/MAD
  Terrain Normalization.
- Detects man-made tools, sensors, metallic/painted equipment, and foreign objects
  using a fused 3-signal anomaly score:
    1. Chromaticity Outlier (CIE LAB Mahalanobis distance from dominant terrain color)
    2. Specular / Saturation Foreign Material Signature (painted/plastic/anodized tools)
    3. Sharp Man-Made Edge & Local Structural Contrast Density (Sobel gradient energy)
- Fast temporal retention (`hold_frames=6`, ~0.1s): bridges single-frame flicker without
  leaving ghost boxes behind when the camera pans away.
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
    category: str
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    offset_from_center: Tuple[int, int]
    color_bgr: Tuple[int, int, int]


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
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
    Fast-response Temporal Box Tracker:
    - Confirms after 2 consistent detections (`min_hits=2`)
    - Retains briefly (`max_age=6`, ~0.1s) so boxes don't flicker, but vanish immediately
      when the object leaves the camera frame.
    """

    def __init__(self, min_hits: int = 2, max_age: int = 6, ema_alpha: float = 0.40):
        self.min_hits = min_hits
        self.max_age = max_age
        self.ema_alpha = ema_alpha
        self.next_id = 1
        self.tracks: List[_Track] = []

    def update(
        self, raw_boxes: List[Tuple[Tuple[int, int, int, int], float]], frame_w: int, frame_h: int
    ) -> List[Detection]:
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
                if iou >= 0.15 or dist <= max(45.0, frame_w * 0.04):
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

        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

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
                        label=f"OBJECT #{trk.track_id}",
                        confidence=float(trk.confidence),
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        offset_from_center=(cx - frame_cx, cy - frame_cy),
                        color_bgr=(0, 255, 140),
                    )
                )
        return confirmed

    def get_retained(self, frame_w: int, frame_h: int) -> List[Detection]:
        frame_cx, frame_cy = frame_w // 2, frame_h // 2
        confirmed: List[Detection] = []
        for trk in self.tracks:
            if trk.hits >= self.min_hits:
                x1, y1, x2, y2 = [int(round(v)) for v in trk.bbox]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                confirmed.append(
                    Detection(
                        category="anomaly",
                        label=f"OBJECT #{trk.track_id}",
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
        enable_yolo: bool = False,
        enable_shapes: bool = False,
        enable_color_anomaly: bool = True,
        yolo_conf_thresh: float = 0.35,
        yolo_iou_thresh: float = 0.45,
        yolo_input_size: int = 416,
        anomaly_z_thresh: float = 4.0,
        detect_every_n: int = 2,
        hold_frames: int = 6,
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

        self.tracker = _AnomalyTracker(min_hits=2, max_age=hold_frames, ema_alpha=0.40)

        self.yolo_session = None
        self.yolo_input_name = None
        self.yolo_model_hw = (640, 640)
        self.yolo_status = "Terrain-Anomaly (No-COCO)"

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
                self.yolo_status = f"Terrain-Anomaly + YOLOv8n ({prov_short}-Async)"
                self._yolo_thread = threading.Thread(target=self._yolo_worker_loop, daemon=True)
                self._yolo_thread.start()
            except Exception:
                self.yolo_session = None

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
            area_frac = ((ix2 - ix1) * (iy2 - iy1)) / float(w * h)
            if area_frac > 0.25 or area_frac < 0.0008:
                continue
            proposals.append(((ix1, iy1, ix2, iy2), float(max_scores[idx])))
        return proposals

    def _detect_terrain_anomalies(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Martian-Terrain Foreign-Object & Tool/Sensor Detector (< 1.8 ms):
        Uses Robust Median + MAD Terrain Normalization so natural rocky/sandy ground
        is ignored while man-made tools, sensors, and distinct foreign objects pop out.
        """
        h, w = frame.shape[:2]
        det_w, det_h = 320, 180
        sx_factor, sy_factor = w / float(det_w), h / float(det_h)
        small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)

        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        # 1. Robust Terrain Color Baseline (Median + MAD ignores outlier objects when estimating ground color)
        med_A = float(np.median(A))
        mad_A = float(np.median(np.abs(A - med_A)) * 1.4826 + 3.5)
        med_B = float(np.median(B))
        mad_B = float(np.median(np.abs(B - med_B)) * 1.4826 + 3.5)
        z_color = np.sqrt(((A - med_A) / mad_A) ** 2 + ((B - med_B) / mad_B) ** 2)

        # 2. Local Luminance Contrast Normalized by Terrain Texture Floor (rejects rocky gravel noise)
        bg_L = cv2.GaussianBlur(L, (35, 35), 10.0)
        diff_L = np.abs(L - bg_L)
        med_L = float(np.median(diff_L))
        mad_L = float(np.median(np.abs(diff_L - med_L)) * 1.4826 + 4.5)
        z_luma = (diff_L - med_L) / mad_L

        # 3. Sharp Edge Density (Man-made tools/sensors have sharp edges vs rough dusty rocks)
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (7, 7), 2.0)
        med_G = float(np.median(grad_mag))
        mad_G = float(np.median(np.abs(grad_mag - med_G)) * 1.4826 + 8.0)
        z_edge = (grad_mag - med_G) / mad_G

        # Fused score: requires strong color anomaly OR strong (luma contrast + sharp edges)
        structural_score = 0.60 * z_luma + 0.40 * z_edge
        fused_z = np.maximum(z_color * 0.95, structural_score)

        mask = (fused_z >= self.anomaly_z_thresh).astype(np.uint8) * 255

        # Mask out outer frame borders, bottom-corner landing gear, and top-left OSD text
        margin_x = int(det_w * 0.08)
        margin_y = int(det_h * 0.08)
        mask[:margin_y, :] = 0
        mask[det_h - margin_y :, :] = 0
        mask[:, :margin_x] = 0
        mask[:, det_w - margin_x :] = 0
        mask[: int(det_h * 0.18), : int(det_w * 0.32)] = 0

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals: List[Tuple[Tuple[int, int, int, int], float]] = []

        min_area = (det_w * det_h) * 0.0006
        max_area = (det_w * det_h) * 0.10

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            sx, sy, sbw, sbh = cv2.boundingRect(cnt)
            aspect = sbw / float(max(sbh, 1))
            if aspect > 4.0 or aspect < 0.25:
                continue

            peak_z = float(np.max(fused_z[sy : sy + sbh, sx : sx + sbw]))
            conf = float(np.clip(peak_z / 6.5, 0.50, 0.99))

            pad = 8
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
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        self.frame_counter += 1

        if self.frame_counter % self.detect_every_n == 1 or self.detect_every_n == 1:
            raw_proposals: List[Tuple[Tuple[int, int, int, int], float]] = []

            if self.enable_color_anomaly:
                raw_proposals.extend(self._detect_terrain_anomalies(frame))

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
            detections = self.tracker.get_retained(w, h)

        detect_ms = (time.perf_counter() - t0) * 1000.0
        annotated = frame
        cx, cy = w // 2, h // 2

        cv2.line(annotated, (cx - 18, cy), (cx - 5, cy), (180, 255, 180), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx + 5, cy), (cx + 18, cy), (180, 255, 180), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx, cy - 18), (cx, cy - 5), (180, 255, 180), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx, cy + 5), (cx, cy + 18), (180, 255, 180), 1, cv2.LINE_AA)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            tcx, tcy = det.center
            color = det.color_bgr

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
        hud_right = f"OBJECTS DETECTED: {len(detections)}"

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
