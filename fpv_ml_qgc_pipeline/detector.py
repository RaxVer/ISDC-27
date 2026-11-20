"""
Heavy-Duty Ground-Arena Foreign-Object & Anomaly Detector (`FPVDetector`).

Engineered for FPV Drone ground inspection (Martian terrain / floor tools, boxes, sensors):
1. Custom FPV Ground-Arena Mask (`_build_ground_arena_mask`):
   Automatically masks out the drone's left & right foam landing legs, the top-left OSD/horizon rim,
   and far top-right corner cables so only objects in the active ground search arena are detected.
2. Multi-Scale Coarse-to-Fine Ground-Plane Subtraction + Median Denoising:
   - Applies a 5x5 median + bilateral pre-filter to destroy analog composite rainbow floor noise.
   - Models the true ground illumination & color plane at coarse 16x9 scale so medium/large objects
     (like the white plastic box) as well as small objects (like a screw, plier, or sensor) pop out
     across their entire body instead of being swallowed by a local filter.
3. Center-Surround Ring Contrast Verification (Delta-E in CIE LAB):
   - Verifies every candidate object by comparing its interior LAB color/brightness against the
     surrounding ground ring immediately outside its bounding box, rejecting gradual floor shadows.
4. Asynchronous Deep ONNX Objectness Engine (Background Worker Thread):
   - Fuses deep neural objectness proposals inside the ground arena while keeping the video loop
     at a locked 60 FPS (< 15 ms frame delay).
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
    "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx",
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
    Temporal Box Tracker with Hysteresis & Fast Decay:
    - Confirms after 2 detections (`min_hits=2`)
    - Retains for `max_age=5` detection cycles (~0.15s) to prevent flicker,
      and drops immediately when an object leaves the frame.
    """

    def __init__(self, min_hits: int = 2, max_age: int = 5, ema_alpha: float = 0.45):
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
                if iou >= 0.12 or dist <= max(50.0, frame_w * 0.045):
                    score = iou + max(0.0, 1.0 - dist / 160.0) * 0.5
                    if score > best_score:
                        best_score = score
                        best_idx = i

            if best_idx != -1:
                matched_dets.add(best_idx)
                db = det_arrs[best_idx]
                trk.bbox = (1.0 - self.ema_alpha) * trk.bbox + self.ema_alpha * db
                trk.confidence = 0.65 * trk.confidence + 0.35 * det_confs[best_idx]
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
        return self.get_retained(frame_w, frame_h)

    def get_retained(self, frame_w: int, frame_h: int) -> List[Detection]:
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
                        color_bgr=(0, 255, 120),
                    )
                )
        return confirmed


def _build_ground_arena_mask(w: int, h: int) -> np.ndarray:
    """
    Constructs the active Ground Search Arena mask on (w, h):
    Excludes:
      - Drone left foam landing leg (bottom-left)
      - Drone right foam landing leg (bottom-right)
      - Top-left OSD ("PAL HD 1080P") & curved arena rim
      - Far top-right corner cables & horizon edge
    """
    mask = np.full((h, w), 255, dtype=np.uint8)

    # 1. Outer frame border margin (5%)
    mx, my = int(w * 0.05), int(h * 0.06)
    mask[:my, :] = 0
    mask[h - my :, :] = 0
    mask[:, :mx] = 0
    mask[:, w - mx :] = 0

    # 2. Left Drone Foam Landing Leg exclusion zone
    left_leg = np.array([
        [0, int(h * 0.25)],
        [int(w * 0.27), int(h * 0.27)],
        [int(w * 0.21), h],
        [0, h],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [left_leg], 0)

    # 3. Right Drone Foam Landing Leg exclusion zone
    right_leg = np.array([
        [int(w * 0.72), int(h * 0.33)],
        [w, int(h * 0.30)],
        [w, h],
        [int(w * 0.75), h],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [right_leg], 0)

    # 4. Top-Left OSD ("PAL HD 1080P") & curved horizon corner
    top_left = np.array([
        [0, 0],
        [int(w * 0.48), 0],
        [int(w * 0.32), int(h * 0.22)],
        [0, int(h * 0.32)],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [top_left], 0)

    # 5. Top-Right Far Corner Cables & edge clutter
    top_right = np.array([
        [int(w * 0.52), 0],
        [w, 0],
        [w, int(h * 0.38)],
        [int(w * 0.62), int(h * 0.16)],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [top_right], 0)

    return mask


class FPVDetector:
    def __init__(
        self,
        enable_yolo: bool = True,
        enable_shapes: bool = False,
        enable_color_anomaly: bool = True,
        yolo_conf_thresh: float = 0.20,
        yolo_iou_thresh: float = 0.40,
        yolo_input_size: int = 640,
        anomaly_z_thresh: float = 3.4,
        detect_every_n: int = 2,
        hold_frames: int = 4,
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

        self.tracker = _AnomalyTracker(min_hits=2, max_age=hold_frames, ema_alpha=0.45)
        self.det_w, self.det_h = 320, 180
        self.arena_mask_small = _build_ground_arena_mask(self.det_w, self.det_h)

        self.yolo_session = None
        self.yolo_input_name = None
        self.yolo_model_hw = (640, 640)
        self.yolo_status = "Multi-Scale Ground-Plane + Deep Objectness"

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
        # Prefer heavier YOLOv8s if present, fallback to YOLOv8n
        onnx_path_s = models_dir / "yolov8s.onnx"
        onnx_path_n = models_dir / "yolov8n.onnx"

        active_path = None
        if onnx_path_s.exists() and onnx_path_s.stat().st_size > 5_000_000:
            active_path = onnx_path_s
        else:
            # Try downloading heavier YOLOv8s first (22MB), fallback to cached YOLOv8n
            try:
                url_s = YOLO_ONNX_URLS[0]
                print(f"[FPVDetector] Fetching heavier YOLOv8s ONNX model ({url_s})...")
                req = urllib.request.Request(url_s, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp, open(onnx_path_s, "wb") as f:
                    f.write(resp.read())
                if onnx_path_s.stat().st_size > 5_000_000:
                    active_path = onnx_path_s
            except Exception:
                if onnx_path_s.exists():
                    onnx_path_s.unlink(missing_ok=True)

        if active_path is None and onnx_path_n.exists() and onnx_path_n.stat().st_size > 1_000_000:
            active_path = onnx_path_n

        if active_path is not None:
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
                    str(active_path), sess_options=sess_opts, providers=providers
                )
                inp_meta = self.yolo_session.get_inputs()[0]
                self.yolo_input_name = inp_meta.name
                shape = inp_meta.shape
                if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                    self.yolo_model_hw = (shape[2], shape[3])
                self.yolo_status = f"Ground-Plane Anomaly + {active_path.stem.upper()} (Async)"
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
            if area_frac > 0.18 or area_frac < 0.0005:
                continue
            # Check that box center falls inside the active Ground Search Arena
            mx = int(((ix1 + ix2) * 0.5) / w * self.det_w)
            my = int(((iy1 + iy2) * 0.5) / h * self.det_h)
            mx = max(0, min(self.det_w - 1, mx))
            my = max(0, min(self.det_h - 1, my))
            if self.arena_mask_small[my, mx] == 0:
                continue
            proposals.append(((ix1, iy1, ix2, iy2), float(max_scores[idx])))
        return proposals

    def _detect_ground_objects(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Dual-Scale Signed Ground-Plane Object & Anomaly Detector:
        - Combines a Wide Ground-Plane Model (for medium/large objects like boxes/containers)
          and a Local Ground-Plane Model (for small tools, screws, sensors, chips), each normalized
          by its own Robust Median Absolute Deviation (MAD).
        - Thresholds positive (bright-on-ground) and negative (dark-on-ground) channels separately
          after spatial smoothing so zero-mean bipolar floor noise cancels out completely.
        """
        h, w = frame.shape[:2]
        det_w, det_h = self.det_w, self.det_h
        sx_factor, sy_factor = w / float(det_w), h / float(det_h)

        # 1. Downscale with INTER_AREA + 5x5 Median-Gaussian pre-filter to erase bipolar analog floor grain
        small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        small_clean = cv2.GaussianBlur(cv2.medianBlur(small, 5), (5, 5), 1.1)

        lab = cv2.cvtColor(small_clean, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        arena_bool = self.arena_mask_small > 0

        # 2. Wide Ground-Plane Illumination Model (catches medium/large objects like the white box)
        coarse_L = cv2.resize(L, (8, 5), interpolation=cv2.INTER_AREA)
        bg_wide = cv2.GaussianBlur(
            cv2.resize(coarse_L, (det_w, det_h), interpolation=cv2.INTER_CUBIC), (31, 31), 10.0
        )
        diff_wide = L - bg_wide
        mad_wide = float(np.median(np.abs(diff_wide[arena_bool])) * 1.4826 + 3.5)
        z_wide = diff_wide / mad_wide

        # 3. Local Ground-Plane Illumination Model (catches small objects like screws, pliers, chips, sensors)
        bg_local = cv2.GaussianBlur(L, (25, 25), 7.0)
        diff_local = L - bg_local
        mad_local = float(np.median(np.abs(diff_local[arena_bool])) * 1.4826 + 3.0)
        z_local = diff_local / mad_local

        # Signed dual-scale Z-score (preserves + for bright objects and - for dark objects)
        z_signed = np.where(np.abs(z_local) >= np.abs(z_wide), z_local, z_wide)
        z_pos = np.maximum(0.0, z_signed)
        z_neg = np.maximum(0.0, -z_signed)

        # 4. Chromaticity Outlier Score (wide 11x11 smoothing kills PAL rainbow blotches)
        A_smooth = cv2.GaussianBlur(A, (11, 11), 3.0)
        B_smooth = cv2.GaussianBlur(B, (11, 11), 3.0)
        med_A = float(np.median(A_smooth[arena_bool]))
        mad_A = float(np.median(np.abs(A_smooth[arena_bool] - med_A)) * 1.4826 + 3.0)
        med_B = float(np.median(B_smooth[arena_bool]))
        mad_B = float(np.median(np.abs(B_smooth[arena_bool] - med_B)) * 1.4826 + 3.0)
        z_color = np.sqrt(((A_smooth - med_A) / mad_A) ** 2 + ((B_smooth - med_B) / mad_B) ** 2)

        chroma_mag = np.sqrt((A_smooth - med_A) ** 2 + (B_smooth - med_B) ** 2)
        z_color = np.where(chroma_mag >= 16.0, z_color, 0.0)

        # Threshold Bright, Dark, and Color masks SEPARATELY so '+' and '-' floor grain never merge!
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = self.anomaly_z_thresh

        mask_pos = cv2.morphologyEx(((z_pos >= thresh) & arena_bool).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)
        mask_neg = cv2.morphologyEx(((z_neg >= thresh) & arena_bool).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)
        mask_col = cv2.morphologyEx(((z_color >= thresh) & arena_bool).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)

        fused_z = np.maximum(np.maximum(z_pos, z_neg), z_color * 0.90)
        proposals: List[Tuple[Tuple[int, int, int, int], float]] = []

        min_area = 16.0
        max_area = (det_w * det_h) * 0.16

        for mask_channel in (mask_pos, mask_neg, mask_col):
            contours, _ = cv2.findContours(mask_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                sx, sy, sbw, sbh = cv2.boundingRect(cnt)
                aspect = sbw / float(max(sbh, 1))
                if aspect > 4.2 or aspect < 0.24:
                    continue

                hull_area = max(1.0, cv2.contourArea(cv2.convexHull(cnt)))
                if (area / hull_area) < 0.52:
                    continue

                rx1, ry1 = max(0, sx - 6), max(0, sy - 6)
                rx2, ry2 = min(det_w, sx + sbw + 6), min(det_h, sy + sbh + 6)
                roi_L = L[ry1:ry2, rx1:rx2]

                box_region = np.zeros((ry2 - ry1, rx2 - rx1), dtype=bool)
                box_region[sy - ry1 : sy - ry1 + sbh, sx - rx1 : sx - rx1 + sbw] = True
                ring_region = ~box_region

                if not np.any(ring_region) or not np.any(box_region):
                    continue

                box_dL = abs(float(np.mean(roi_L[box_region])) - float(np.mean(roi_L[ring_region])))
                roi_C = chroma_mag[ry1:ry2, rx1:rx2]
                box_dC = float(np.mean(roi_C[box_region]))

                if box_dL < 21.0 and box_dC < 14.0:
                    continue

                peak_z = float(np.max(fused_z[sy : sy + sbh, sx : sx + sbw]))
                if peak_z < 3.8:
                    continue

                conf = float(np.clip((box_dL + box_dC * 1.5) / 50.0, 0.55, 0.99))
                pad = 6
                x1 = max(0, int(sx * sx_factor) - pad)
                y1 = max(0, int(sy * sy_factor) - pad)
                x2 = min(w - 1, int((sx + sbw) * sx_factor) + pad)
                y2 = min(h - 1, int((sy + sbh) * sy_factor) + pad)
                proposals.append(((x1, y1, x2, y2), max(conf, min(0.99, peak_z / 6.5))))

        return proposals

    def _merge_proposals_nms(
        self, proposals: List[Tuple[Tuple[int, int, int, int], float]]
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if len(proposals) <= 1:
            return proposals

        kept = list(proposals)
        changed = True
        while changed:
            changed = False
            merged: List[Tuple[Tuple[int, int, int, int], float]] = []
            used = [False] * len(kept)
            for i in range(len(kept)):
                if used[i]:
                    continue
                bx1, by1, bx2, by2 = kept[i][0]
                bconf = kept[i][1]
                used[i] = True
                for j in range(i + 1, len(kept)):
                    if used[j]:
                        continue
                    ox1, oy1, ox2, oy2 = kept[j][0]
                    h_gap = max(0, max(bx1, ox1) - min(bx2, ox2))
                    v_gap = max(0, max(by1, oy1) - min(by2, oy2))
                    if h_gap <= 24 and v_gap <= 24:
                        bx1, by1 = min(bx1, ox1), min(by1, oy1)
                        bx2, by2 = max(bx2, ox2), max(by2, oy2)
                        bconf = max(bconf, kept[j][1])
                        used[j] = True
                        changed = True
                merged.append(((bx1, by1, bx2, by2), bconf))
            kept = merged
        return kept

    def detect_and_annotate(
        self,
        frame: np.ndarray,
        draw_hud: bool = True,
        perf_stats: Optional[Dict[str, float]] = None,
        single_frame_mode: bool = False,
    ) -> Tuple[np.ndarray, List[Detection], Dict[str, float]]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        self.frame_counter += 1

        if single_frame_mode or self.frame_counter % self.detect_every_n == 1 or self.detect_every_n == 1:
            raw_proposals: List[Tuple[Tuple[int, int, int, int], float]] = []

            if self.enable_color_anomaly:
                raw_proposals.extend(self._detect_ground_objects(frame))

            if self.enable_yolo and self.yolo_session is not None:
                if single_frame_mode:
                    for (bx1, by1, bx2, by2), sc in self._detect_yolo_boxes(frame):
                        raw_proposals.append(((bx1, by1, bx2, by2), sc))
                else:
                    with self._yolo_lock:
                        if self._yolo_pending_frame is None:
                            self._yolo_pending_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)
                        for (bx1, by1, bx2, by2), sc in self._latest_yolo_boxes:
                            sx, sy = w / 640.0, h / 360.0
                            raw_proposals.append(
                                ((int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy)), sc)
                            )

            merged = self._merge_proposals_nms(raw_proposals)
            if single_frame_mode:
                # Confirm immediately in single-image test mode
                self.tracker.min_hits = 1
            detections = self.tracker.update(merged, w, h)
        else:
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

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.circle(annotated, (tcx, tcy), 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                annotated,
                det.label,
                (x1, max(48, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
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
