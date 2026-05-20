import ctypes
import ctypes.wintypes as wt
from dataclasses import dataclass
import struct
import sys
from typing import Any

from loguru import logger
from PyMemoryEditor import OpenProcess
from PyMemoryEditor.process.abstract import AbstractProcess
from PyMemoryEditor.process.errors import ProcessNotFoundError
from src.utils.exceptions import FieldResolveError
from src.utils.exceptions import MemoryReadError

from .game_ptrs import PLAYER_PTR


# Copyright 2023 MihaiStreames
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


PROC_QUERY_INFO = 0x0400
PROC_VM_READ = 0x0010
WIN_FALSE = 0


@dataclass
class MemoryState:
    """Snapshot of game RAM fields at a given moment."""

    ypos: float
    xpos: float
    hp: int
    gems: int
    ammo: int
    gem_high: int
    combo: int


@dataclass(frozen=True)
class AttachedMemory:
    """
    Active validated connection to a running process.

    Invariant: both ``_proc`` and ``_module_base`` are guaranteed valid for the lifetime of this object.
    Do not use standalone; use ``MemoryReader.attach()``.
    """

    _proc: AbstractProcess
    _module_base: int

    def _read_ptr(self, addr: int) -> int:
        data: bytes = self._proc.read_process_memory(addr, bytes, 4)
        logger.trace(f"reading ptr {struct.unpack_from('<I', data)[0]}")
        return struct.unpack_from("<I", data)[0]

    def _read_typed(self, addr: int, type_str: str) -> float:
        size = 4 if type_str == "float" else 8
        logger.trace(f"reading typed {self._proc.read_process_memory(addr, float, size)} ({size})")
        return self._proc.read_process_memory(addr, float, size)

    def _get_ptr_addr(self, base: int, offsets: list[int]) -> int:
        addr = self._read_ptr(base)

        for offset in offsets[:-1]:
            addr = self._read_ptr(addr + offset)

        return addr + offsets[-1]

    def _get_field(self, field: str, module_base: int) -> float:
        entry: Any = PLAYER_PTR[field]
        type_str: str = entry["type"]

        if "bases" in entry:
            bases: list[int] = entry["bases"]
            offsets_list: list[list[int]] = entry["offsets"]
        else:
            bases: list[int] = [entry["base"]]
            offsets_list: list[list[int]] = [entry["offsets"]]

        for base, offsets in zip(bases, offsets_list, strict=False):
            try:
                addr = self._get_ptr_addr(module_base + base, offsets)
                return self._read_typed(addr, type_str)
            except MemoryReadError:
                continue

        raise FieldResolveError

    def read(self) -> MemoryState | None:
        """Sample current game state. Raises ``MemoryReadError`` if process dies."""
        # no guards needed; if we have AttachedMemory it means we're attached
        return MemoryState(
            ypos=float(self._get_field("ypos", self._module_base)),
            xpos=float(self._get_field("xpos", self._module_base)),
            hp=int(self._get_field("hp", self._module_base)),
            gems=int(self._get_field("gems", self._module_base)),
            ammo=int(self._get_field("ammo", self._module_base)),
            gem_high=int(self._get_field("gem_high", self._module_base)),
            combo=int(self._get_field("combo", self._module_base)),
        )

    def close(self) -> None:
        """Terminate session. This object must not be used after calling this."""
        logger.debug(f"closed {self._proc._process_info.process_name}")  # noqa: SLF001 (readability)
        self._proc.close()


def _resolve_module_base_win(pid: int, proc_name: str) -> int | None:
    if sys.platform != "win32":
        return None

    handle = ctypes.windll.kernel32.OpenProcess(PROC_QUERY_INFO | PROC_VM_READ, WIN_FALSE, pid)
    if not handle:
        return None

    try:
        name_buf = ctypes.create_unicode_buffer(512)

        modules = (wt.HMODULE * 1024)()
        needed = wt.DWORD()

        ctypes.windll.psapi.EnumProcessModules(handle, modules, ctypes.sizeof(modules), ctypes.byref(needed))
        count = needed.value // ctypes.sizeof(wt.HMODULE)

        for mod in modules[:count]:
            ctypes.windll.psapi.GetModuleBaseNameW(handle, mod, name_buf, 260)
            if proc_name.lower() == name_buf.value.lower():
                return mod
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)

    return None


def _resolve_module_base_linux(proc: AbstractProcess, proc_name: str) -> int | None:
    for region in proc.get_memory_regions():
        path: bytes = region["struct"].Path or b""
        if proc_name.encode() in path:
            return region["address"]

    return None


def _resolve_module_base(proc: AbstractProcess, proc_name: str) -> int | None:
    if sys.platform == "win32":
        return _resolve_module_base_win(proc.pid, proc_name)
    if sys.platform == "linux":
        return _resolve_module_base_linux(proc, proc_name)

    return None


# TODO @Sincos: consts.py (for proc_name)
def attach(proc_name: str = "Downwell.exe") -> AttachedMemory | None:
    """
    Attempt to find and attach to a process.

    Returns an ``AttachedMemory`` if successful, ``None`` if process isn't running or module base can't be resolved.
    """
    try:
        proc = OpenProcess(process_name=proc_name)
    except ProcessNotFoundError:
        return None

    base = _resolve_module_base(proc, proc_name)
    if base is None:
        logger.warning(f"attached to {proc_name} but module base not found in memory")
        proc.close()
        return None

    logger.debug(f"attached to {proc_name} (base 0x{base:x})")
    return AttachedMemory(proc, base)
