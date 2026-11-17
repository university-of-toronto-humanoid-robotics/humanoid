"""The checksum Unitree's robots expect on /lowcmd. The G1 ignores a command whose crc field is wrong.

Works on anything shaped like unitree_hg/msg/LowCmd, so there is no ROS in here.
"""
import struct

import numpy as np

N_MOTORS = 35   # slots in the message; the G1 uses the first 29
# The command as the robot's C struct: mode_pr, mode_machine, 35 motors (mode, q, dq, tau, kp, kd, reserve), then
# 4 reserved words. The crc field follows and isn't part of the sum.
_LAYOUT = struct.Struct('<2B2x' + 'B3x5fI' * N_MOTORS + '4I')
_POLYNOMIAL = 0x04C11DB7


def _table():
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ _POLYNOMIAL if crc & 0x80000000 else crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return table


_TABLE = _table()


def crc(msg):
    """The crc for a LowCmd: CRC-32/MPEG-2 over the struct's 32-bit words, most significant bit first."""
    fields = [msg.mode_pr, msg.mode_machine]
    for m in msg.motor_cmd:
        fields += [m.mode, m.q, m.dq, m.tau, m.kp, m.kd, m.reserve]
    fields += [int(r) for r in msg.reserve]
    data = np.frombuffer(_LAYOUT.pack(*fields), dtype='<u4').astype('>u4').tobytes()
    value = 0xFFFFFFFF
    for byte in data:
        value = ((value << 8) & 0xFFFFFFFF) ^ _TABLE[(value >> 24) ^ byte]
    return value
