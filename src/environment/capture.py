import threading

import numpy as np
from mss import mss


class ScreenCapture:
    """MSS-based capture"""

    def __init__(self):
        self._thread_local = threading.local()
        self.monitor = None
        self._last_bbox = None

    def _get_sct(self):
        """Get or create MSS instance for current thread"""
        if not hasattr(self._thread_local, "sct"):
            self._thread_local.sct = mss()
        return self._thread_local.sct

    def set_region(self, left, top, width, height):
        """Configure capture region"""
        self.monitor = {"top": top, "left": left, "width": width, "height": height}
        self._last_bbox = (left, top, width, height)

    def capture(self):
        """Capture and return as numpy array (H, W, C)"""
        if self.monitor is None:
            raise RuntimeError("Must call set_region() first")

        # Get thread-specific MSS instance
        sct = self._get_sct()
        sct_img = sct.grab(self.monitor)
        img = np.array(sct_img, dtype=np.uint8)
        return img[:, :, [2, 1, 0]]  # BGR -> RGB

    def __del__(self):
        """Cleanup"""
        if hasattr(self, "_thread_local") and hasattr(self._thread_local, "sct"):
            try:
                self._thread_local.sct.close()
            except:
                pass
