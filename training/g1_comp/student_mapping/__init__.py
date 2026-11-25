"""ROS-like height observations for a future shared student policy.

The expert tasks are unchanged. ``attach_student_map`` accepts a copied RPL
configuration and replaces only its actor terrain-scan observation.
"""

from .observation import StudentMapCfg, StudentHeightMap, attach_student_map

__all__ = ["StudentMapCfg", "StudentHeightMap", "attach_student_map"]
