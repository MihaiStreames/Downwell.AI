import sys


if sys.platform == "win32":
    from .direct_input import DirectInput as Input  # noqa: F401
else:
    from .pynput_input import PynputInput as Input  # noqa: F401
