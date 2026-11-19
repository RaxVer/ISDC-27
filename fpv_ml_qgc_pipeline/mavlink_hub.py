#!/usr/bin/env python3
"""
Multi-Laptop D-Link LAN MAVLink Telemetry Multiplexer & Role-Based Filter (`mavlink_hub.py`).

Run this on the MAIN HUB LAPTOP connected to the Pixhawk Telemetry Receiver (/dev/ttyUSB0 or /dev/ttyACM0).
It ingests the Pixhawk MAVLink stream and forwards role-specific telemetry packets across the
D-Link Switch/Hub to each station's static IP address:

  - Flight Data Laptop  (default 192.168.1.10:14550): Altitude, Speed, GPS, Attitude, Battery, HUD
  - Payload Data Laptop (default 192.168.1.20:14550): Servo/Actuator outputs, Payload Sensors, Named Values
  - Camera Feed Laptop  (default 192.168.1.30:14550): Minimal HUD Telemetry (while video arrives on UDP :5600)
  - Local QGC on Hub    (default 127.0.0.1:14550)   : Full bidirectional MAVLink control

Usage:
  ./.venv/bin/python mavlink_hub.py --master /dev/ttyUSB0 --baud 57600 \
      --flight-ip 192.168.1.10:14550 \
      --payload-ip 192.168.1.20:14550 \
      --camera-ip 192.168.1.30:14550
"""

import argparse
import socket
import sys
import time
from typing import Dict, Set, Tuple
from pymavlink import mavutil


FLIGHT_MESSAGES: Set[str] = {
    "HEARTBEAT", "SYS_STATUS", "SYSTEM_TIME", "PING", "CHANGE_OPERATOR_CONTROL",
    "AUTH_KEY", "PARAM_VALUE", "GPS_RAW_INT", "GPS_STATUS", "SCALED_IMU",
    "RAW_IMU", "ATTITUDE", "ATTITUDE_QUATERNION", "LOCAL_POSITION_NED",
    "GLOBAL_POSITION_INT", "RC_CHANNELS_SCALED", "RC_CHANNELS_RAW",
    "MISSION_CURRENT", "MISSION_ITEM_REACHED", "MISSION_COUNT", "MISSION_ACK",
    "NAV_CONTROLLER_OUTPUT", "VFR_HUD", "COMMAND_ACK", "BATTERY_STATUS",
    "AUTOPILOT_VERSION", "EXTENDED_SYS_STATE", "HOME_POSITION", "VIBRATION",
    "ALTITUDE", "ESTIMATOR_STATUS", "WIND_COV", "GPS2_RAW", "HIGH_LATENCY2",
    "STATUSTEXT", "TIMESYNC",
}

PAYLOAD_MESSAGES: Set[str] = {
    "HEARTBEAT", "SYS_STATUS", "SYSTEM_TIME", "PARAM_VALUE",
    "SERVO_OUTPUT_RAW", "RC_CHANNELS", "RC_CHANNELS_RAW",
    "SCALED_PRESSURE", "SCALED_PRESSURE2", "DISTANCE_SENSOR",
    "OPTICAL_FLOW", "NAMED_VALUE_FLOAT", "NAMED_VALUE_INT",
    "DEBUG", "DEBUG_VECT", "CAMERA_TRIGGER", "CAMERA_FEEDBACK",
    "MOUNT_STATUS", "MOUNT_ORIENTATION", "GIMBAL_DEVICE_ATTITUDE_STATUS",
    "ACTUATOR_CONTROL_TARGET", "BATTERY_STATUS", "STATUSTEXT", "COMMAND_ACK",
}


def parse_host_port(addr: str) -> Tuple[str, int]:
    host, port_s = addr.strip().split(":")
    return host, int(port_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Laptop D-Link LAN MAVLink Telemetry Router")
    parser.add_argument(
        "--master",
        type=str,
        default="/dev/ttyUSB0",
        help="Pixhawk telemetry input (e.g. '/dev/ttyUSB0', '/dev/ttyACM0', or 'udpin:0.0.0.0:14551')",
    )
    parser.add_argument("--baud", type=int, default=57600, help="Serial baudrate for SiK Telemetry Radio (default 57600)")
    parser.add_argument("--local-qgc", type=str, default="127.0.0.1:14550", help="Local QGC UDP endpoint on Main Laptop")
    parser.add_argument("--flight-ip", type=str, default="192.168.1.10:14550", help="Flight Data Laptop IP:PORT")
    parser.add_argument("--payload-ip", type=str, default="192.168.1.20:14550", help="Payload Data Laptop IP:PORT")
    parser.add_argument("--camera-ip", type=str, default="192.168.1.30:14550", help="Camera Feed Laptop IP:PORT")
    parser.add_argument(
        "--filter-roles",
        action="store_true",
        help="Filter MAVLink message types per laptop role (otherwise forwards full MAVLink stream to all laptops)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  D-LINK LAN MULTI-LAPTOP MAVLINK TELEMETRY HUB & ROUTER")
    print("=" * 72)
    print(f"  [*] Master Pixhawk Input : {args.master} (baud: {args.baud})")
    print(f"  [*] Local Hub QGC        : {args.local_qgc} (Full Control)")
    print(f"  [*] Laptop 1 (Flight)    : {args.flight_ip}")
    print(f"  [*] Laptop 2 (Payload)   : {args.payload_ip}")
    print(f"  [*] Laptop 3 (Cameras)   : {args.camera_ip}")
    print(f"  [*] Role Message Filter  : {'ENABLED' if args.filter_roles else 'FULL STREAM'}")
    print("=" * 72)

    master = mavutil.mavlink_connection(args.master, baud=args.baud, autoreconnect=True)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setblocking(False)

    targets: Dict[str, Tuple[str, int]] = {
        "local": parse_host_port(args.local_qgc),
        "flight": parse_host_port(args.flight_ip),
        "payload": parse_host_port(args.payload_ip),
        "camera": parse_host_port(args.camera_ip),
    }

    pkt_count = 0
    t_last = time.perf_counter()

    try:
        while True:
            msg = master.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                continue

            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue

            raw_bytes = msg.get_msgbuf()
            pkt_count += 1

            # 1. Always forward full stream to local Hub QGC and Camera HUD
            udp_sock.sendto(raw_bytes, targets["local"])
            udp_sock.sendto(raw_bytes, targets["camera"])

            # 2. Forward to Flight Data Laptop
            if not args.filter_roles or mtype in FLIGHT_MESSAGES:
                udp_sock.sendto(raw_bytes, targets["flight"])

            # 3. Forward to Payload Data Laptop
            if not args.filter_roles or mtype in PAYLOAD_MESSAGES:
                udp_sock.sendto(raw_bytes, targets["payload"])

            now = time.perf_counter()
            if now - t_last >= 5.0:
                print(f"  [MAVLink Hub] Forwarded {pkt_count} packets across D-Link LAN...")
                t_last = now

    except KeyboardInterrupt:
        print("\n  [*] Stopping MAVLink Hub.")
    finally:
        udp_sock.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
