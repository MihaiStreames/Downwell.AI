import sys


if sys.platform == "win32":
    from .pymem_memory import PymemMemory as Memory  # noqa: F401
else:
    from .null_memory import NullMemory as Memory  # noqa: F401
