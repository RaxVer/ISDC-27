#!/usr/bin/env python3
"""
End-to-End Real-Time FPV Drone Video Enhancement (CNN+RNN), Multi-Modal Detection
(YOLOv8 + Geometric Shapes + Spectral Color Anomalies), and Zero-Latency H.264
GStreamer/RTP Output Streamer to QGroundControl.

Usage Examples:
  1. Run with Live Analog FPV Drone Simulator directly into QGroundControl (UDP 5600):
     ./.venv/bin/python pipeline.py --source sim

  2. Run with physical USB Capture Card (/dev/video0) from Zeus 800mW VTX Receiver:
     ./.venv/bin/python pipeline.py --source /dev/video0

  3. Run Dual FPV Cameras (/dev/video0 and /dev/video2) Side-by-Side into QGroundControl:
     ./.venv/bin/python pipeline.py --source /dev/video0 --source2 /dev/video2 --dual-mode sbs

  4. Stream over D-Link LAN Hub to a remote Camera Station Laptop (e.g. 192.168.1.30):
     ./.venv/bin/python pipeline.py --source /dev/video0 --udp-host 192.168.1.30 --udp-port 5600
"""

import argparse
import signal
import sys
import time
from typing import List, Optional
import cv2
import numpy as np

from capture import FPVVideoSource
from detector import FPVDetector
from enhancer import FPVEnhancer
from streamer import QGCStreamer


def make_split_comparison(raw_frame: np.ndarray, enhanced_annotated: np.ndarray) -> np.ndarray:
    """
    Creates a split-screen wipe comparison:
      Left 40%: Raw upsampled unenhanced analog VTX feed
      Right 60%: CNN+RNN Super-Resolved + Multi-Modal AI Detection feed
    """
    eh, ew = enhanced_annotated.shape[:2]
    raw_up = cv2.resize(raw_frame, (ew, eh), interpolation=cv2.INTER_NEAREST)
    split_x = int(ew * 0.38)
    out = enhanced_annotated.copy()
    out[34:, :split_x] = raw_up[34:, :split_x]
    cv2.line(out, (split_x, 34), (split_x, eh), (0, 255, 255), 2)
    cv2.rectangle(out, (10, eh - 36), (175, eh - 10), (15, 15, 15), -1)
    cv2.putText(out, "RAW ANALOG VTX", (16, eh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (split_x + 10, eh - 36), (split_x + 250, eh - 10), (15, 15, 15), -1)
    cv2.putText(
        out,
        "CNN+RNN 2x SR + AI DETECT",
        (split_x + 16, eh - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 160),
        1,
        cv2.LINE_AA,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time FPV Video Enhancement (CNN+RNN) + YOLO/Shape/Anomaly Detection -> QGroundControl Streamer"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="sim",
        help="Primary video source: '/dev/video0', '0', video file path, or 'sim' for Analog FPV Simulator",
    )
    parser.add_argument(
        "--source2",
        type=str,
        default=None,
        help="Optional secondary video source for Dual FPV Camera mode (e.g. '/dev/video2' or 'sim')",
    )
    parser.add_argument(
        "--dual-mode",
        type=str,
        default="sbs",
        choices=["sbs", "pip", "cam1", "cam2"],
        help="Dual camera layout inside QGC: 'sbs' (Side-by-Side), 'pip' (Picture-in-Picture), 'cam1', 'cam2'",
    )
    parser.add_argument("--input-width", type=int, default=1280, help="Capture width (default: 1280 for 720p)")
    parser.add_argument("--input-height", type=int, default=720, help="Capture height (default: 720 for 720p)")
    parser.add_argument(
        "--upscale",
        type=float,
        default=1.5,
        help="Super-resolution factor: 1.5 (default: 720p -> 1080p Full HD), 2.0 (720p -> 1440p QHD), or 1.0",
    )
    parser.add_argument(
        "--enhancer-mode",
        type=str,
        default="cnn_rnn",
        choices=["cnn_rnn", "fast_hybrid", "passthrough"],
        help="Enhancement engine mode: 'cnn_rnn' (ESPCN + Temporal RNN), 'fast_hybrid', or 'passthrough'",
    )
    parser.add_argument("--fps", type=int, default=60, help="Target output stream framerate (default: 60)")
    parser.add_argument("--udp-host", type=str, default="127.0.0.1", help="Target QGroundControl IP (default: 127.0.0.1)")
    parser.add_argument("--udp-port", type=int, default=5600, help="Target QGroundControl UDP Video Port (default: 5600)")
    parser.add_argument(
        "--extra-targets",
        type=str,
        default="",
        help="Comma-separated additional IP:PORT targets across the D-Link LAN (e.g. '192.168.1.30:5600')",
    )
    parser.add_argument(
        "--stream-format",
        type=str,
        default="rtp",
        choices=["rtp", "mpegts"],
        help="QGC stream protocol: 'rtp' (QGC 'UDP h.264 Video Stream') or 'mpegts' (QGC 'MPEG-TS (h.264)')",
    )
    parser.add_argument("--bitrate", type=int, default=5500, help="H.264 stream bitrate in kbps (default: 5500)")
    parser.add_argument("--yolo", action="store_true", help="Also include COCO YOLOv8n box proposals")
    parser.add_argument("--no-anomaly", action="store_true", help="Disable terrain foreign-object/anomaly detection")
    parser.add_argument("--split-view", action="store_true", help="Show Raw vs Enhanced split comparison")
    parser.add_argument("--stream-qgc", action="store_true", help="Also stream H.264 RTP to QGroundControl (UDP 5600)")
    parser.add_argument("--no-preview", action="store_true", help="Disable local full-quality preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run continuously)")
    parser.add_argument("--save-sample", type=str, default="", help="Optional path to save a sample output PNG frame")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    show_preview = not args.no_preview

    out_w = int(round(args.input_width * args.upscale))
    out_h = int(round(args.input_height * args.upscale))
    out_w += out_w % 2
    out_h += out_h % 2

    print("=" * 78)
    print("  FPV DRONE ML ENHANCEMENT & TERRAIN ANOMALY DETECTION PIPELINE")
    print("=" * 78)

    video_source = FPVVideoSource(
        source1=args.source,
        source2=args.source2,
        width=args.input_width,
        height=args.input_height,
        fps=args.fps,
        dual_mode=args.dual_mode,
    )

    enhancer = FPVEnhancer(
        upscale_factor=args.upscale,
        mode=args.enhancer_mode,
        target_width=out_w,
        target_height=out_h,
    )

    detector = FPVDetector(
        enable_yolo=args.yolo,
        enable_shapes=False,
        enable_color_anomaly=not args.no_anomaly,
    )

    streamer = None
    if args.stream_qgc:
        extra_list: List[str] = [t.strip() for t in args.extra_targets.split(",") if t.strip()]
        streamer = QGCStreamer(
            width=out_w,
            height=out_h,
            fps=args.fps,
            host=args.udp_host,
            port=args.udp_port,
            extra_targets=extra_list,
            stream_format=args.stream_format,
            bitrate_kbps=args.bitrate,
        )

    print(f"  [*] Video Source      : {video_source.source_description} ({args.input_width}x{args.input_height} @ {args.fps}fps)")
    print(f"  [*] ML Enhancer       : {enhancer.backend_name} | Mode: {enhancer.mode} -> {out_w}x{out_h}")
    print(f"  [*] Detection Engine  : {detector.yolo_status} | Anomaly={'ON' if detector.enable_color_anomaly else 'OFF'}")
    print(f"  [*] Local Display     : {'FULL ' + str(out_w) + 'x' + str(out_h) if show_preview else 'DISABLED'}")
    print(f"  [*] QGC Video Output  : {streamer.backend_used if streamer else 'DISABLED (Local Mode — add --stream-qgc to enable)'}")
    print("=" * 78)

    running = True

    def _sig_handler(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    frame_count = 0
    fps_ema = float(args.fps)
    e2e_delay_ema = 8.0
    split_view = args.split_view
    frame_interval = 1.0 / max(1, args.fps)
    sample_saved = False

    if show_preview:
        win_name = f"FPV ML Pipeline ({out_w}x{out_h}) [e=Mode y=YOLO c=Detect v=Split q=Quit]"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(out_w, 1600), min(out_h, 900))

    try:
        while running:
            t_loop = time.perf_counter()
            ok, raw_frame = video_source.read()
            if not ok or raw_frame is None:
                time.sleep(0.005)
                continue

            t_proc_start = time.perf_counter()

            # Stage 1: Crisp Neural Detail/Super-Resolution + Mild Temporal RNN
            enhanced_frame, enh_metrics = enhancer.enhance(raw_frame)

            # Stage 2: Martian-Terrain Foreign-Object / Anomaly Detection
            perf_stats = {
                "fps": fps_ema,
                "enhance_ms": enh_metrics["enhance_ms"],
                "e2e_delay_ms": e2e_delay_ema,
                "enhancer_mode": enhancer.mode,
            }
            annotated_frame, detections, det_metrics = detector.detect_and_annotate(
                enhanced_frame, draw_hud=True, perf_stats=perf_stats
            )

            output_frame = (
                make_split_comparison(raw_frame, annotated_frame)
                if split_view
                else annotated_frame
            )

            # Stage 3: Optional QGroundControl H.264 RTP Stream
            if streamer is not None:
                streamer.write(output_frame)

            proc_delay_ms = (time.perf_counter() - t_proc_start) * 1000.0
            e2e_delay_ema = 0.85 * e2e_delay_ema + 0.15 * proc_delay_ms
            frame_count += 1

            if args.save_sample and not sample_saved and frame_count >= 5:
                cv2.imwrite(args.save_sample, output_frame)
                print(f"  [+] Saved sample output frame to: {args.save_sample}")
                sample_saved = True

            if show_preview:
                cv2.imshow(win_name, output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("e"):
                    modes = ["cnn_rnn", "fast_hybrid", "passthrough"]
                    enhancer.mode = modes[(modes.index(enhancer.mode) + 1) % len(modes)]
                elif key == ord("y"):
                    detector.enable_yolo = not detector.enable_yolo
                elif key == ord("c"):
                    detector.enable_color_anomaly = not detector.enable_color_anomaly
                elif key == ord("v"):
                    split_view = not split_view
                elif key == ord("1"):
                    video_source.dual_mode = "cam1"
                elif key == ord("2"):
                    video_source.dual_mode = "cam2"
                elif key == ord("b"):
                    video_source.dual_mode = "sbs"

            # Maintain smooth real-time cadence for RTP stream
            sleep_rem = frame_interval - (time.perf_counter() - t_loop)
            if sleep_rem > 0.001:
                time.sleep(sleep_rem)

            dt_loop = time.perf_counter() - t_loop
            inst_fps = 1.0 / max(dt_loop, 1e-4)
            fps_ema = 0.85 * fps_ema + 0.15 * inst_fps

            if frame_count % 60 == 0:
                print(
                    f"  [Stream Active] Frame {frame_count:05d} | {fps_ema:4.1f} FPS | "
                    f"Total Delay: {e2e_delay_ema:4.1f} ms (<120ms OK) "
                    f"[SR: {enh_metrics['enhance_ms']:4.1f}ms, Det: {det_metrics['detect_ms']:4.1f}ms] | "
                    f"Targets: {det_metrics['count']}"
                )

            if 0 < args.max_frames <= frame_count:
                break

    finally:
        detector.close()
        if streamer is not None:
            streamer.close()
        video_source.release()
        if show_preview:
            cv2.destroyAllWindows()
        print("  [*] Pipeline stopped cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
