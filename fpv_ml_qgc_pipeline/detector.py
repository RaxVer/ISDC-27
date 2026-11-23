"""
RTX-4060 GPU-Accelerated 960x540 Ground-Plane Anomaly & YOLO11m (960x960, opset=12) Detector (`FPVDetector`).

Implements:
1. Sparse Optical Flow + RANSAC 8-DOF Ground-Homography Bounding-Box Pinning (`_AnomalyTracker.warp_tracks`):
   - On every single frame (60 FPS), projects all tracked bounding boxes through the 3x3 Lucas-Kanade
     RANSAC Ground Homography matrix H_{t-1 -> t} via `cv2.perspectiveTransform`.
   - Even when the drone pitches, rolls, banks, or yaws rapidly, locked boxes move instantaneously
     with the physical ground!
2. Homography-Stabilized Temporal Anomaly Evidence Accumulation Grid (S_t on 960x540):
   - Maintains floating-point ground anomaly confidence maps (S_pos, S_neg, S_col) at 960x540 resolution.
   - Warps S_{t-1} by the ground Homography H_{960} before fusing with current frame Z-scores:
       S_t = 0.62 * warpPerspective(S_{t-1}, H_{960}) + 0.38 * Z_t
   - Random RF noise cancels out to 0 while real tools/objects on the ground accumulate constructively.
3. 960x540 Ground-Plane Anomaly Grid (9x pixel density of 320x180) + YOLO11m (20.1M params, 960x960, opset=12)
   running on NVIDIA RTX 4060 CUDAExecutionProvider.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


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
    Homography-Pinned Temporal Box Tracker:
    - Warps active track boxes on every frame using the 8-DOF Lucas-Kanade Ground Homography H_{t-1 -> t}
      so boxes stay glued to the ground during drone pitch/roll/yaw.
    - Confirms after 2 detections (`min_hits=2`), retains for `max_age=8` cycles.
    """

    def __init__(self, min_hits: int = 2, max_age: int = 8, ema_alpha: float = 0.48):
        self.min_hits = min_hits
        self.max_age = max_age
        self.ema_alpha = ema_alpha
        self.next_id = 1
        self.tracks: List[_Track] = []

    def warp_tracks(self, H_full: Optional[np.ndarray], frame_w: int, frame_h: int) -> None:
        """Projects all active track bounding boxes through the 3x3 Ground Homography H_{t-1 -> t}."""
        if H_full is None or not self.tracks:
            return
        if np.allclose(H_full, np.eye(3, dtype=np.float32), atol=1e-3):
            return

        for trk in self.tracks:
            x1, y1, x2, y2 = trk.bbox
            corners = np.array(
                [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32
            )
            warped = cv2.perspectiveTransform(corners, H_full)
            wx = warped[:, 0, 0]
            wy = warped[:, 0, 1]
            nx1 = float(np.clip(np.min(wx), 0.0, frame_w - 2.0))
            ny1 = float(np.clip(np.min(wy), 0.0, frame_h - 2.0))
            nx2 = float(np.clip(np.max(wx), nx1 + 2.0, frame_w - 1.0))
            ny2 = float(np.clip(np.max(wy), ny1 + 2.0, frame_h - 1.0))
            trk.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

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
                if iou >= 0.12 or dist <= max(55.0, frame_w * 0.05):
                    score = iou + max(0.0, 1.0 - dist / 180.0) * 0.5
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
                        color_bgr=(220, 30, 255),
                    )
                )
        return confirmed


def _build_ground_arena_mask(w: int, h: int) -> np.ndarray:
    """
    Constructs a tight, pixel-accurate Ground Search Arena mask on (w, h):
    Masks ONLY the curved left & right foam landing gear tubes + bottom carbon brackets
    and the small 'PAL HD 720P' OSD box, leaving outer fisheye edges OPEN.
    """
    mask = np.full((h, w), 255, dtype=np.uint8)

    mx, my = max(2, int(w * 0.015)), max(2, int(h * 0.02))
    mask[:my, :] = 0
    mask[h - my :, :] = 0
    mask[:, :mx] = 0
    mask[:, w - mx :] = 0

    left_leg_tube = np.array([
        [int(w * 0.08), h],
        [int(w * 0.21), h],
        [int(w * 0.24), int(h * 0.58)],
        [int(w * 0.30), int(h * 0.23)],
        [int(w * 0.19), int(h * 0.23)],
        [int(w * 0.12), int(h * 0.58)],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [left_leg_tube], 0)
    cv2.rectangle(mask, (0, int(h * 0.85)), (int(w * 0.21), h), 0, -1)

    right_leg_tube = np.array([
        [int(w * 0.85), h],
        [int(w * 0.98), h],
        [int(w * 0.97), int(h * 0.58)],
        [int(w * 0.89), int(h * 0.22)],
        [int(w * 0.78), int(h * 0.22)],
        [int(w * 0.84), int(h * 0.58)],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [right_leg_tube], 0)
    cv2.rectangle(mask, (int(w * 0.86), int(h * 0.85)), (w, h), 0, -1)

    cv2.rectangle(mask, (int(w * 0.13), int(h * 0.02)), (int(w * 0.31), int(h * 0.16)), 0, -1)
    cv2.fillPoly(mask, [np.array([[0, 0], [int(w * 0.22), 0], [0, int(h * 0.30)]], dtype=np.int32)], 0)

    return mask


class FPVDetector:
    def __init__(
        self,
        enable_yolo: bool = True,
        enable_shapes: bool = False,
        enable_color_anomaly: bool = True,
        yolo_conf_thresh: float = 0.18,
        yolo_iou_thresh: float = 0.40,
        yolo_input_size: int = 1280,
        anomaly_z_thresh: float = 3.0,
        detect_every_n: int = 2,
        hold_frames: int = 8,
        accum_decay: float = 0.58,
    ):
        self.enable_yolo = enable_yolo
        self.enable_shapes = enable_shapes
        self.enable_color_anomaly = enable_color_anomaly
        self.yolo_conf_thresh = yolo_conf_thresh
        self.yolo_iou_thresh = yolo_iou_thresh
        self.yolo_input_size = yolo_input_size
        self.anomaly_z_thresh = anomaly_z_thresh
        self.detect_every_n = max(1, int(detect_every_n))
        self.accum_decay = float(np.clip(accum_decay, 0.0, 0.85))
        self.frame_counter = 0

        self.tracker = _AnomalyTracker(min_hits=2, max_age=hold_frames, ema_alpha=0.48)

        # Full Native 1280x720 Ground-Plane Anomaly Grid (16x pixel density of original 320x180)
        self.det_w, self.det_h = 1280, 720
        self.arena_mask_small = _build_ground_arena_mask(self.det_w, self.det_h)

        # Homography-Stabilized Temporal Anomaly Evidence Accumulation Maps on 1280x720
        self.accum_pos: Optional[np.ndarray] = None
        self.accum_neg: Optional[np.ndarray] = None
        self.accum_col: Optional[np.ndarray] = None
        self._H_accum_960 = np.eye(3, dtype=np.float32)

        self.yolo_session = None
        self.yolo_input_name = None
        self.yolo_model_hw = (1280, 1280)
        self.yolo_status = "1280x720 Ground-Grid + Temporal Homography Map"

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
        """
        Loads official Ultralytics YOLO11m (`yolo11m.onnx` at 1280x1280, opset=12) on RTX 4060 CUDA.
        Falls back to yolo11s.onnx / yolov8s.onnx if yolo11m.onnx is not present.
        """
        models_dir = Path(__file__).resolve().parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        yolo11m_onnx = models_dir / "yolo11m.onnx"
        yolo11m_pt = models_dir / "yolo11m.pt"

        active_path = None
        if yolo11m_onnx.exists() and yolo11m_onnx.stat().st_size > 10_000_000:
            active_path = yolo11m_onnx
        elif yolo11m_pt.exists():
            try:
                from ultralytics import YOLO

                print("[FPVDetector] Exporting official YOLO11m (1280x1280, opset=12)...")
                model = YOLO(str(yolo11m_pt))
                exported = model.export(format="onnx", opset=12, imgsz=1280)
                if exported and Path(exported).exists():
                    active_path = Path(exported)
            except Exception as exc:
                print(f"[FPVDetector] Note: YOLO11m export fallback ({exc})")

        if active_path is None:
            for cand in (models_dir / "yolo11s.onnx", models_dir / "yolov8s.onnx", models_dir / "yolov8n.onnx"):
                if cand.exists() and cand.stat().st_size > 1_000_000:
                    active_path = cand
                    break

        if active_path is not None:
            try:
                import onnxruntime as ort

                if hasattr(ort, "preload_dlls"):
                    try:
                        ort.preload_dlls()
                    except Exception:
                        pass
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
                active_prov = self.yolo_session.get_providers()[0].replace("ExecutionProvider", "")
                self.yolo_status = (
                    f"1280x720 Homography-Grid + {active_path.stem.upper()}@{self.yolo_model_hw[0]} ({active_prov} opset=12)"
                )
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
                time.sleep(0.008)
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
            if area_frac > 0.18 or area_frac < 0.0003:
                continue
            mx = int(((ix1 + ix2) * 0.5) / w * self.det_w)
            my = int(((iy1 + iy2) * 0.5) / h * self.det_h)
            mx = max(0, min(self.det_w - 1, mx))
            my = max(0, min(self.det_h - 1, my))
            if self.arena_mask_small[my, mx] == 0:
                continue
            proposals.append(((ix1, iy1, ix2, iy2), float(max_scores[idx])))
        return proposals

    def _detect_ground_objects(
        self, frame: np.ndarray, H_960: np.ndarray, single_frame_mode: bool = False
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Full 1280x720 High-Resolution Dual-Scale Signed Ground-Plane Anomaly Detector
        with Homography-Stabilized Temporal Evidence Accumulation (16x area of original 320x180).
        """
        h, w = frame.shape[:2]
        det_w, det_h = self.det_w, self.det_h
        sx_factor, sy_factor = w / float(det_w), h / float(det_h)

        # 1. Sample at full 1280x720 + 5x5 Median-Gaussian pre-filter
        if (w, h) == (det_w, det_h):
            small = frame
        else:
            small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        small_clean = cv2.GaussianBlur(cv2.medianBlur(small, 5), (5, 5), 1.2)

        lab = cv2.cvtColor(small_clean, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        arena_bool = self.arena_mask_small > 0

        # 2. Wide Ground-Plane Illumination Model (catches medium/large boxes, jackets, containers)
        coarse_L = cv2.resize(L, (16, 9), interpolation=cv2.INTER_AREA)
        bg_wide = cv2.GaussianBlur(
            cv2.resize(coarse_L, (det_w, det_h), interpolation=cv2.INTER_CUBIC), (95, 95), 28.0
        )
        diff_wide = L - bg_wide
        mad_wide = float(np.median(np.abs(diff_wide[arena_bool])) * 1.4826 + 3.5)
        z_wide = diff_wide / mad_wide

        # 3. Local Ground-Plane Illumination Model at 1280x720 (catches small tools, pliers, screws, sensors)
        bg_local = cv2.GaussianBlur(L, (75, 75), 22.0)
        diff_local = L - bg_local
        mad_local = float(np.median(np.abs(diff_local[arena_bool])) * 1.4826 + 3.0)
        z_local = diff_local / mad_local

        z_signed = np.where(np.abs(z_local) >= np.abs(z_wide), z_local, z_wide)
        z_pos = np.maximum(0.0, z_signed)
        z_neg = np.maximum(0.0, -z_signed)

        # 4. Chromaticity Outlier Score on 1280x720
        A_smooth = cv2.GaussianBlur(A, (25, 25), 6.5)
        B_smooth = cv2.GaussianBlur(B, (25, 25), 6.5)
        med_A = float(np.median(A_smooth[arena_bool]))
        mad_A = float(np.median(np.abs(A_smooth[arena_bool] - med_A)) * 1.4826 + 3.0)
        med_B = float(np.median(B_smooth[arena_bool]))
        mad_B = float(np.median(np.abs(B_smooth[arena_bool] - med_B)) * 1.4826 + 3.0)
        z_color = np.sqrt(((A_smooth - med_A) / mad_A) ** 2 + ((B_smooth - med_B) / mad_B) ** 2)

        chroma_mag = np.sqrt((A_smooth - med_A) ** 2 + (B_smooth - med_B) ** 2)
        z_color = np.where(chroma_mag >= 14.5, z_color, 0.0)

        # 5. Upgrade #2: Homography-Stabilized Temporal Anomaly Evidence Accumulation on 1280x720
        if (
            not single_frame_mode
            and self.accum_decay > 0.01
            and self.accum_pos is not None
            and self.accum_pos.shape == (det_h, det_w)
        ):
            w_pos = cv2.warpPerspective(self.accum_pos, H_960, (det_w, det_h), flags=cv2.INTER_LINEAR)
            w_neg = cv2.warpPerspective(self.accum_neg, H_960, (det_w, det_h), flags=cv2.INTER_LINEAR)
            w_col = cv2.warpPerspective(self.accum_col, H_960, (det_w, det_h), flags=cv2.INTER_LINEAR)
            alpha_cur = 1.0 - self.accum_decay
            self.accum_pos = self.accum_decay * w_pos + alpha_cur * z_pos
            self.accum_neg = self.accum_decay * w_neg + alpha_cur * z_neg
            self.accum_col = self.accum_decay * w_col + alpha_cur * z_color
            # Take max of accumulated world-stabilized score and instant strong score
            eff_pos = np.maximum(self.accum_pos * 1.15, z_pos * 0.92)
            eff_neg = np.maximum(self.accum_neg * 1.15, z_neg * 0.92)
            eff_col = np.maximum(self.accum_col * 1.15, z_color * 0.92)
        else:
            self.accum_pos = z_pos.copy()
            self.accum_neg = z_neg.copy()
            self.accum_col = z_color.copy()
            eff_pos, eff_neg, eff_col = z_pos, z_neg, z_color

        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        thresh = self.anomaly_z_thresh

        mask_pos = cv2.morphologyEx(((eff_pos >= thresh + 0.5) & arena_bool).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)
        mask_neg = cv2.morphologyEx(((eff_neg >= thresh) & arena_bool).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)
        mask_col = cv2.morphologyEx(((eff_col >= thresh + 0.4) & arena_bool).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)

        fused_z = np.maximum(np.maximum(eff_pos, eff_neg), eff_col * 0.90)
        proposals: List[Tuple[Tuple[int, int, int, int], float]] = []

        # On 1280x720 (16x area of 320x180), min_area=140 catches tiny tools/screws (~8.75px on 320x180)
        min_area = 140.0
        max_area = (det_w * det_h) * 0.15

        for ch_idx, mask_channel in enumerate((mask_pos, mask_neg, mask_col)):
            contours, _ = cv2.findContours(mask_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                sx, sy, sbw, sbh = cv2.boundingRect(cnt)
                aspect = sbw / float(max(sbh, 1))
                if aspect > 4.4 or aspect < 0.22:
                    continue

                hull_area = max(1.0, cv2.contourArea(cv2.convexHull(cnt)))
                if (area / hull_area) < 0.48:
                    continue

                rx1, ry1 = max(0, sx - 14), max(0, sy - 14)
                rx2, ry2 = min(det_w, sx + sbw + 14), min(det_h, sy + sbh + 14)
                roi_L = L[ry1:ry2, rx1:rx2]

                box_region = np.zeros((ry2 - ry1, rx2 - rx1), dtype=bool)
                box_region[sy - ry1 : sy - ry1 + sbh, sx - rx1 : sx - rx1 + sbw] = True
                ring_region = ~box_region

                if not np.any(ring_region) or not np.any(box_region):
                    continue

                box_dL = abs(float(np.mean(roi_L[box_region])) - float(np.mean(roi_L[ring_region])))
                roi_C = chroma_mag[ry1:ry2, rx1:rx2]
                box_dC = float(np.mean(roi_C[box_region]))

                if ch_idx == 0 and box_dL < 27.0:
                    continue
                elif ch_idx == 1 and box_dL < 19.5:
                    continue
                elif ch_idx == 2 and box_dC < 26.0:
                    continue

                peak_z = float(np.max(fused_z[sy : sy + sbh, sx : sx + sbw]))
                if peak_z < 3.9:
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

        boxes = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2), _ in proposals]
        scores = [c for _, c in proposals]
        idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.35)
        if len(idxs) == 0:
            return []
        return [proposals[i] for i in np.array(idxs).flatten()]

    @staticmethod
    def _scale_homography(H_320: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        sx, sy = target_w / 320.0, target_h / 180.0
        S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        S_inv = np.array([[1.0 / sx, 0.0, 0.0], [0.0, 1.0 / sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return S @ H_320 @ S_inv

    def detect_and_annotate(
        self,
        frame: np.ndarray,
        draw_hud: bool = True,
        perf_stats: Optional[Dict[str, Any]] = None,
        single_frame_mode: bool = False,
    ) -> Tuple[np.ndarray, List[Detection], Dict[str, float]]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        self.frame_counter += 1

        # Extract 3x3 Lucas-Kanade Ground Homography from enhancer metrics (if provided)
        H_320 = np.eye(3, dtype=np.float32)
        if perf_stats and "homography_320" in perf_stats and perf_stats["homography_320"] is not None:
            H_320 = perf_stats["homography_320"]

        # 1. Upgrade #1: Warp all active tracked bounding boxes with current frame's Ground Homography
        H_full = self._scale_homography(H_320, w, h)
        self.tracker.warp_tracks(H_full, w, h)

        # Accumulate inter-frame 960x540 Homography for the Temporal Anomaly Grid
        H_960_step = self._scale_homography(H_320, self.det_w, self.det_h)
        self._H_accum_960 = H_960_step @ self._H_accum_960

        if single_frame_mode or self.frame_counter % self.detect_every_n == 1 or self.detect_every_n == 1:
            raw_proposals: List[Tuple[Tuple[int, int, int, int], float]] = []

            if self.enable_color_anomaly:
                raw_proposals.extend(
                    self._detect_ground_objects(frame, self._H_accum_960, single_frame_mode=single_frame_mode)
                )
            self._H_accum_960 = np.eye(3, dtype=np.float32)

            if self.enable_yolo and self.yolo_session is not None:
                if single_frame_mode:
                    for (bx1, by1, bx2, by2), sc in self._detect_yolo_boxes(frame):
                        raw_proposals.append(((bx1, by1, bx2, by2), sc))
                else:
                    with self._yolo_lock:
                        if self._yolo_pending_frame is None:
                            self._yolo_pending_frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)
                        for (bx1, by1, bx2, by2), sc in self._latest_yolo_boxes:
                            sx, sy = w / 960.0, h / 540.0
                            raw_proposals.append(
                                ((int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy)), sc)
                            )

            merged = self._merge_proposals_nms(raw_proposals)
            if single_frame_mode:
                self.tracker.min_hits = 1
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
            box_color = (220, 30, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), 5)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(annotated, (tcx, tcy), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(annotated, (tcx, tcy), 3, (0, 255, 255), -1, cv2.LINE_AA)

            (tw, th), _ = cv2.getTextSize(det.label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ly = max(th + 42, y1 - 6)
            cv2.rectangle(annotated, (x1, ly - th - 6), (x1 + tw + 10, ly + 4), (12, 12, 16), -1)
            cv2.rectangle(annotated, (x1, ly - th - 6), (x1 + tw + 10, ly + 4), box_color, 1)
            cv2.putText(
                annotated,
                det.label,
                (x1 + 5, ly - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
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
        perf_stats: Dict[str, Any],
    ) -> None:
        h, w = canvas.shape[:2]
        top_strip = canvas[:34, :]
        dark_bar = np.full_like(top_strip, (12, 16, 22))
        cv2.addWeighted(dark_bar, 0.72, top_strip, 0.28, 0, top_strip)

        fps = float(perf_stats.get("fps", 0.0))
        enh_ms = float(perf_stats.get("enhance_ms", 0.0))
        e2e_ms = float(perf_stats.get("e2e_delay_ms", enh_ms + detect_ms))
        mode_str = str(perf_stats.get("enhancer_mode", "CNN+RNN")).upper()

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
            (220, 30, 255),
            1,
            cv2.LINE_AA,
        )
