import platform

from loguru import logger
import pymem
import pymem.exception

from src.utils.game_attributes import PLAYER_PTR


class Player:
    def __init__(self, pc: pymem.Pymem, game_module: int):
        self.os = platform.system()
        self.pc = pc
        self.game_module = game_module
        self.attr = PLAYER_PTR

        if self.os != "Windows":
            raise NotImplementedError("Only Windows is supported")

    def is_gem_high(self) -> bool:
        value = self.get_value("gemHigh")
        if value is None:
            logger.debug("Failed to read gemHigh value")
            return False
        return value >= 100

    def get_ptr_addr(self, base: int, offsets: list[int]) -> int:
        try:
            addr = self.pc.read_int(base)
            for offset in offsets[:-1]:
                addr = self.pc.read_int(addr + offset)
            return addr + offsets[-1]
        except pymem.exception.MemoryReadError as e:
            raise e

    def get_type(self, attr_type: str, address: int) -> float | None:
        try:
            return getattr(self.pc, f"read_{attr_type}")(address)  # type: ignore
        except AttributeError:
            logger.error(f"Unknown type: {attr_type}")
            return None
        except pymem.exception.MemoryReadError as e:
            raise e

    def get_value(self, attribute: str) -> float | None:
        if attribute not in self.attr:
            logger.error(f"Unknown attribute: {attribute}")
            return None

        attr_data = self.attr[attribute]
        if "bases" in attr_data:
            bases: list[int] = attr_data["bases"]
            offsets_list: list[list[int]] = attr_data["offsets"]
        else:
            bases = [attr_data["base"]]
            offsets_list = [attr_data["offsets"]]

        for base, offsets in zip(bases, offsets_list, strict=False):
            try:
                address = self.get_ptr_addr(self.game_module + base, offsets)
                value = self.get_type(attr_data["type"], address)
                return value
            except pymem.exception.MemoryReadError:
                continue

        logger.debug(f"Failed to read {attribute} from all available addresses")
        return None

    def validate_connection(self) -> bool:
        try:
            # Attempt to read a known value to check connection
            self.get_value("hp")
            return True
        except Exception as e:
            logger.error(f"Lost connection to game process: {e}")
            return False
