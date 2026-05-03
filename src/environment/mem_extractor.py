import sys


if sys.platform != "win32":
    raise NotImplementedError("Only Windows is supported.")

from typing import Any
from typing import cast

from loguru import logger
import pymem
import pymem.exception

from src.utils.consts import GEM_HIGH_THRESHOLD
from src.utils.game_attributes import PLAYER_PTR


class Player:
    def __init__(self, pc: pymem.Pymem, game_module: int) -> None:
        self._pc: pymem.Pymem = pc
        self._game_module: int = game_module
        self._attr: dict[str, dict[str, object]] = PLAYER_PTR

    def is_gem_high(self) -> bool:
        value = self.get_value("gemHigh")
        if value is None:
            logger.trace("Failed to read gemHigh value")
            return False
        return value >= GEM_HIGH_THRESHOLD

    def _get_ptr_addr(self, base: int, offsets: list[int]) -> int:
        try:
            addr = int(self._pc.read_int(base))
            for offset in offsets[:-1]:
                addr = int(self._pc.read_int(addr + offset))
            return addr + offsets[-1]

        except pymem.exception.MemoryReadError as e:
            raise e

    def _get_type(self, attr_type: str, address: int) -> float | None:
        try:
            return getattr(self._pc, f"read_{attr_type}")(address)
        except AttributeError:
            logger.error(f"Unknown type: {attr_type}")
            return None
        except pymem.exception.MemoryReadError as e:
            raise e

    def get_value(self, attribute: str) -> float | None:
        if attribute not in self._attr:
            logger.error(f"Unknown attribute: {attribute}")
            return None

        attr_data = cast(dict[str, Any], self._attr[attribute])
        if "bases" in attr_data:
            bases: list[int] = cast(list[int], attr_data["bases"])
            offsets_list: list[list[int]] = cast(list[list[int]], attr_data["offsets"])
        else:
            bases = [cast(int, attr_data["base"])]
            offsets_list = [cast(list[int], attr_data["offsets"])]

        for base, offsets in zip(bases, offsets_list, strict=False):
            try:
                address = self._get_ptr_addr(self._game_module + base, offsets)
                value = self._get_type(cast(str, attr_data["type"]), address)
                return value
            except pymem.exception.MemoryReadError:
                continue

        logger.trace(f"Failed to read {attribute} from all available addresses")
        return None
