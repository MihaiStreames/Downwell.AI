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


from dataclasses import dataclass
import struct
from typing import cast

from loguru import logger
from PyMemoryEditor import OpenProcess
from PyMemoryEditor.process.abstract import AbstractProcess
from PyMemoryEditor.process.errors import ProcessNotFoundError
from src.consts import PROCESS_NAME
from src.utils.exceptions import FieldResolveError

from ._module_base import resolve_module_base
from .game_ptrs import PLAYER_PTR


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

    Invariant: both ``_process`` and ``_module_base`` are guaranteed valid for the lifetime of this object.
    Do not use standalone; use ``attach()``.
    """

    _process: AbstractProcess
    _module_base: int

    def _read_ptr(self, addr: int) -> int | None:
        # PyMemoryEditor does not raise on bad reads, returns garbage on failure
        data: bytes = self._process.read_process_memory(addr, bytes, 4)
        result = struct.unpack_from("<I", data)[0]
        logger.trace(f"reading ptr {result:#x}")
        return result if result != 0 else None

    def _read_typed(self, addr: int, type_str: str) -> float:
        size = 4 if type_str == "float" else 8
        value: float = self._process.read_process_memory(addr, float, size)
        logger.trace(f"reading typed {value} ({size}b)")
        return value

    def _get_ptr_addr(self, base: int, offsets: list[int]) -> int | None:
        addr = self._read_ptr(base)
        if addr is None:
            return None

        for offset in offsets[:-1]:
            addr = self._read_ptr(addr + offset)
            if addr is None:
                return None

        return addr + offsets[-1]

    def _get_field(self, field: str, module_base: int) -> float:
        entry: dict[str, object] = PLAYER_PTR[field]
        type_str: str = str(entry["type"])

        if "bases" in entry:
            bases: list[int] = cast("list[int]", entry["bases"])
            offsets_list: list[list[int]] = cast("list[list[int]]", entry["offsets"])
        else:
            bases = [cast("int", entry["base"])]
            offsets_list = [cast("list[int]", entry["offsets"])]

        for base, offsets in zip(bases, offsets_list, strict=False):
            addr = self._get_ptr_addr(module_base + base, offsets)
            if addr is None:
                continue

            return self._read_typed(addr, type_str)

        raise FieldResolveError(field)

    def read(self) -> MemoryState:
        """Sample current game state. Raises ``FieldResolveError`` if all chains fail."""
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
        logger.debug(f"closed {self._process._process_info.process_name}")  # noqa: SLF001 (readability)
        self._process.close()


def attach(proc_name: str = PROCESS_NAME) -> AttachedMemory | None:
    """
    Attempt to find and attach to a process. Works on both Windows (via ``OpenProcess``) and Linux (via ``/proc/``).

    Returns an ``AttachedMemory`` if successful, ``None`` if process isn't running or module base can't be resolved.
    """
    try:
        proc = OpenProcess(process_name=proc_name)
    except ProcessNotFoundError:
        return None

    base = resolve_module_base(proc, proc_name)
    if base is None:
        logger.error(f"attached to {proc_name} but module base not found in memory")
        proc.close()
        return None

    logger.debug(f"attached to {proc_name} (base 0x{base:x})")
    return AttachedMemory(proc, base)
