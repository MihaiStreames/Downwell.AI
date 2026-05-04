import sys


if sys.platform == "win32":
    from .dxcam_capture import DXCamCapture as Capture  # noqa: F401
else:
    from .fast_capture import FastgrabCapture as Capture  # noqa: F401
