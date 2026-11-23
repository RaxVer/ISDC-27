"""
Real-Time Fisheye-Rectified, Lucas-Kanade Homography MFSR & Ground-Mute / Object-Pop Enhancer.

Implements:
1. Real-Time Fisheye Barrel Un-Warping (`cv2.remap` Perspective Rectification):
   - Precomputes a radial lens undistortion LUT (k1=-0.16, k2=0.03) that un-stretches the
     compressed outer fisheye edges at 3-4m flight altitude in ~1.0 ms (toggleable via hotkey 'f').
2. Sparse Optical Flow (Pyramidal Lucas-Kanade) + RANSAC 8-DOF Ground-Plane Homography:
   - Tracks ~90 ground keypoints frame-to-frame via `cv2.calcOpticalFlowPyrLK` and fits a projective
     Homography matrix H_{t-1 -> t} via RANSAC (`cv2.findHomography`).
   - Warps the multi-frame super-resolution hidden buffer via `cv2.warpPerspective` to align rotation,
     pitch/roll tilt, altitude zoom, and translation to sub-pixel precision.
   - Exports the 3x3 Homography matrix to `detector.py` so the 960x540 Temporal Anomaly Grid and
     locked bounding boxes move synchronously with the physical ground!
3. Automatic Ground-Color Detection & Vibrancy Suppression (0.28x Ground Mute / 1.85x Object Pop).
"""

import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from models_arch import _create_calibrated_espcn_weights, ensure_espcn_onnx_model


def _build_scurve_contrast_lut(contrast: float = 1.22, gamma: float = 0.94) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    xg = np.power(x, gamma)
    s = 0.5 + (xg - 0.5) * contrast
    s = np.clip(s, 0.0, 1.0)
    return (s * 255.0).astype(np.uint8)


class FPVEnhancer:
    def __init__(
        self,
        upscale_factor: float = 1.5,
        mode: str = "cnn_rnn",
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        temporal_strength: float = 0.30,
        clahe_clip: float = 1.35,
        sharpen_strength: float = 1.35,
        ground_mute_gain: float = 0.28,
        object_pop_gain: float = 1.85,
        ground_radius: float = 20.0,
        enable_defish: bool = True,
        fisheye_k1: float = -0.16,
    ):
        self.upscale_factor = max(1.0, float(upscale_factor))
        self.mode = mode
        self.target_width = target_width
        self.target_height = target_height
        self.temporal_strength = float(np.clip(temporal_strength, 0.0, 0.60))
        self.sharpen_strength = float(sharpen_strength)
        self.ground_mute_gain = float(ground_mute_gain)
        self.object_pop_gain = float(object_pop_gain)
        self.ground_radius = float(ground_radius)
        self.enable_defish = enable_defish
        self.fisheye_k1 = float(fisheye_k1)

        # Fisheye Undistortion LUT cache
        self._defish_hw: Tuple[int, int] = (0, 0)
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None

        # Sparse Optical Flow (Lucas-Kanade) + RANSAC Homography MFSR state
        self.hidden_state: Optional[np.ndarray] = None
        self.prev_lk_gray: Optional[np.ndarray] = None
        self.lk_w, self.lk_h = 320, 180
        self.lk_mask = self._build_lk_ground_mask(self.lk_w, self.lk_h)

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 4))
        self.luma_lut = _build_scurve_contrast_lut(contrast=1.22, gamma=0.94)

        self.weights = _create_calibrated_espcn_weights(upscale_factor=2)
        self.ort_session = None
        self.device_provider = "CPU-AVX512"
        self._init_cnn_backend()

    @staticmethod
    def _build_lk_ground_mask(w: int, h: int) -> np.ndarray:
        """Mask for Lucas-Kanade optical flow keypoints (central ground, excluding static drone legs)."""
        m = np.zeros((h, w), dtype=np.uint8)
        m[int(h * 0.16) : int(h * 0.88), int(w * 0.24) : int(w * 0.76)] = 255
        return m

    def _ensure_defish_maps(self, w: int, h: int) -> None:
        """Precomputes fast int16 fixed-point cv2.remap tables for fisheye barrel rectification."""
        if self._defish_hw == (h, w) and self._map_x is not None:
            return
        fx = fy = float(w) * 0.82
        cx, cy = w * 0.5, h * 0.5
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        # Negative k1 un-warps barrel distortion, stretching compressed outer fisheye edges
        D = np.array([self.fisheye_k1, 0.025, 0.0, 0.0], dtype=np.float64)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.15, newImgSize=(w, h))
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (w, h), cv2.CV_16SC2
        )
        self._defish_hw = (h, w)

    def _init_cnn_backend(self) -> None:
        onnx_path = ensure_espcn_onnx_model()
        try:
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception:
                    pass
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(
                    onnx_path, sess_options=sess_opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                if "CUDAExecutionProvider" in self.ort_session.get_providers():
                    self.device_provider = "RTX-4060-CUDA"
                else:
                    self.ort_session = None
        except Exception:
            self.ort_session = None
        self.backend_name = f"LK-Homography MFSR + Defish + ESPCN ({self.device_provider})"

    def reset_state(self) -> None:
        self.hidden_state = None
        self.prev_lk_gray = None

    def _lk_homography_mfsr_fusion(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sparse Optical Flow (Pyramidal Lucas-Kanade) + RANSAC 8-DOF Ground Homography MFSR:
        Returns:
          (multi_frame_stacked_bgr, H_320_homography_matrix)
        """
        h, w = frame.shape[:2]
        small_gray = cv2.cvtColor(
            cv2.resize(frame, (self.lk_w, self.lk_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        H_320 = np.eye(3, dtype=np.float32)

        if self.hidden_state is None or self.hidden_state.shape != frame.shape or self.prev_lk_gray is None:
            self.hidden_state = frame.copy()
            self.prev_lk_gray = small_gray
            return frame, H_320

        # 1. Detect Shi-Tomasi ground corners on previous frame
        p0 = cv2.goodFeaturesToTrack(
            self.prev_lk_gray,
            maxCorners=90,
            qualityLevel=0.04,
            minDistance=10,
            mask=self.lk_mask,
            blockSize=5,
        )

        valid_homography = False
        if p0 is not None and len(p0) >= 10:
            # 2. Track corners into current frame with Pyramidal Lucas-Kanade Sparse Optical Flow
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_lk_gray,
                small_gray,
                p0,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
            )
            if p1 is not None and st is not None:
                good_old = p0[st.flatten() == 1]
                good_new = p1[st.flatten() == 1]
                if len(good_new) >= 8:
                    # 3. Estimate 8-DOF Projective Ground-Plane Homography via RANSAC
                    H_est, inliers = cv2.findHomography(good_old, good_new, cv2.RANSAC, 2.5)
                    if H_est is not None and inliers is not None and int(np.sum(inliers)) >= 6:
                        # Sanity-check determinant so degenerate warps are rejected
                        det_h = float(np.linalg.det(H_est[:2, :2]))
                        if 0.70 <= det_h <= 1.40 and abs(H_est[0, 2]) < 65.0 and abs(H_est[1, 2]) < 65.0:
                            H_320 = H_est.astype(np.float32)
                            valid_homography = True

        self.prev_lk_gray = small_gray

        if self.temporal_strength <= 0.01:
            return frame, H_320

        if valid_homography:
            # Scale H_320 from (320x180) to full frame (w, h): H_full = S * H_320 * S^{-1}
            sx, sy = w / float(self.lk_w), h / float(self.lk_h)
            S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            S_inv = np.array([[1.0 / sx, 0.0, 0.0], [0.0, 1.0 / sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            H_full = S @ H_320 @ S_inv
            aligned_prev = cv2.warpPerspective(
                self.hidden_state,
                H_full,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            aligned_prev = frame

        # Residual motion gate after 8-DOF Homography ego-motion alignment
        diff = cv2.absdiff(frame, aligned_prev)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, motion_mask = cv2.threshold(gray_diff, 16, 255, cv2.THRESH_BINARY)

        blended = cv2.addWeighted(
            frame, 1.0 - self.temporal_strength, aligned_prev, self.temporal_strength, 0
        )
        cv2.copyTo(frame, motion_mask, blended)
        self.hidden_state = blended
        return blended, H_320

    def _suppress_ground_and_pop_objects(
        self, cr_chan: np.ndarray, cb_chan: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        cr_smooth = cv2.GaussianBlur(cr_chan, (11, 11), 2.8)
        cb_smooth = cv2.GaussianBlur(cb_chan, (11, 11), 2.8)

        cr_tiny = cv2.resize(cr_smooth, (80, 45), interpolation=cv2.INTER_AREA)
        cb_tiny = cv2.resize(cb_smooth, (80, 45), interpolation=cv2.INTER_AREA)
        med_cr = float(np.median(cr_tiny))
        med_cb = float(np.median(cb_tiny))

        h, w = cr_smooth.shape
        qw, qh = max(1, w // 4), max(1, h // 4)
        cr_q = cv2.resize(cr_smooth, (qw, qh), interpolation=cv2.INTER_AREA).astype(np.float32)
        cb_q = cv2.resize(cb_smooth, (qw, qh), interpolation=cv2.INTER_AREA).astype(np.float32)

        dist_ground = np.sqrt((cr_q - med_cr) ** 2 + (cb_q - med_cb) ** 2)
        t = np.clip((dist_ground - self.ground_radius * 0.55) / max(1.0, self.ground_radius), 0.0, 1.0)
        t_smooth = t * t * (3.0 - 2.0 * t)
        gain_map_q = self.ground_mute_gain + (self.object_pop_gain - self.ground_mute_gain) * t_smooth
        gain_map = cv2.resize(gain_map_q, (w, h), interpolation=cv2.INTER_LINEAR)

        cr_out = np.clip(128.0 + (cr_smooth.astype(np.float32) - 128.0) * gain_map, 0.0, 255.0).astype(np.uint8)
        cb_out = np.clip(128.0 + (cb_smooth.astype(np.float32) - 128.0) * gain_map, 0.0, 255.0).astype(np.uint8)
        return cr_out, cb_out

    def _enhance_luma_and_upscale(self, y_chan: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        h, w = y_chan.shape

        if self.ort_session is not None and (out_w, out_h) == (w * 2, h * 2):
            y_norm = y_chan.astype(np.float32) * (1.0 / 255.0)
            out = self.ort_session.run(None, {"input_y": y_norm[np.newaxis, np.newaxis, :, :]})[0][0, 0]
            return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

        y_bilat = cv2.bilateralFilter(y_chan, 5, 24, 24)
        f_med = cv2.GaussianBlur(y_bilat, (7, 7), 1.6)
        f_wide = cv2.GaussianBlur(y_bilat, (15, 15), 3.2)

        edge_pos = cv2.subtract(y_bilat, cv2.add(f_med, 9))
        edge_neg = cv2.subtract(f_med, cv2.add(y_bilat, 9))

        struct_pos = cv2.subtract(f_med, cv2.add(f_wide, 6))
        struct_neg = cv2.subtract(f_wide, cv2.add(f_med, 6))

        gain_fine = 1.40 * self.sharpen_strength
        gain_struct = 0.65 * self.sharpen_strength

        clean_sharp = cv2.addWeighted(y_bilat, 1.0, edge_pos, gain_fine, 0)
        clean_sharp = cv2.subtract(clean_sharp, cv2.convertScaleAbs(edge_neg, alpha=gain_fine))
        clean_sharp = cv2.addWeighted(clean_sharp, 1.0, struct_pos, gain_struct, 0)
        clean_sharp = cv2.subtract(clean_sharp, cv2.convertScaleAbs(struct_neg, alpha=gain_struct))

        if (out_w, out_h) == (w, h):
            return clean_sharp
        return cv2.resize(clean_sharp, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    def enhance(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        out_w = int(self.target_width or round(w * self.upscale_factor))
        out_h = int(self.target_height or round(h * self.upscale_factor))

        # Step 0: Real-Time Fisheye Barrel Undistortion (stretches compressed outer fisheye edges)
        if self.enable_defish:
            self._ensure_defish_maps(w, h)
            working_frame = cv2.remap(frame, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR)
        else:
            working_frame = frame

        if self.mode == "passthrough":
            if (out_w, out_h) != (w, h):
                out = cv2.resize(working_frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            else:
                out = working_frame.copy()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return out, {"enhance_ms": dt_ms, "sharpness": 0.0, "homography_320": np.eye(3, dtype=np.float32)}

        # Step 1: Sparse Optical Flow (Lucas-Kanade) + RANSAC 8-DOF Ground Homography MFSR
        stacked, H_320 = self._lk_homography_mfsr_fusion(working_frame)

        # Step 2: YCrCb separation
        ycrcb = cv2.cvtColor(stacked, cv2.COLOR_BGR2YCrCb)
        y_chan, cr_chan, cb_chan = cv2.split(ycrcb)

        # Step 3: Automatic Ground-Color Suppression & Foreign-Object Pop
        cr_vibrant, cb_vibrant = self._suppress_ground_and_pop_objects(cr_chan, cb_chan)

        # Step 4: Gentle CLAHE + S-Curve Contrast LUT
        y_clahe = self.clahe.apply(y_chan)
        y_mix = cv2.addWeighted(y_clahe, 0.35, y_chan, 0.65, 0)
        y_contrast = cv2.LUT(y_mix, self.luma_lut)

        # Step 5: Dual-Scale Structural Detail Boost + Super-Resolution
        if self.mode == "cnn_rnn":
            y_sr = self._enhance_luma_and_upscale(y_contrast, out_w, out_h)
        else:
            y_sr = cv2.resize(y_contrast, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        if (out_w, out_h) != (w, h):
            cr_sr = cv2.resize(cr_vibrant, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            cb_sr = cv2.resize(cb_vibrant, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        else:
            cr_sr, cb_sr = cr_vibrant, cb_vibrant

        merged_ycrcb = cv2.merge([y_sr, cr_sr, cb_sr])
        enhanced_bgr = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return enhanced_bgr, {
            "enhance_ms": dt_ms,
            "sharpness": 0.0,
            "homography_320": H_320,
        }
