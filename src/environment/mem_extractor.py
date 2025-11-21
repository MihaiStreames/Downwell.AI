import platform
from typing import Any, cast

from loguru import logger
import pymem  # type: ignore[import-not-found]
import pymem.exception  # type: ignore[import-not-found]

from src.utils.game_attributes import PLAYER_PTR


class Player:
    """Memory reader for Downwell player data.

    Parameters
    ----------
    pc : pymem.Pymem
        Pymem instance attached to the game process.
    game_module : int
        Base address of the game module.

    Attributes
    ----------
    os : str
        Operating system name.
    pc : pymem.Pymem
        Pymem process handle.
    game_module : int
        Game module base address.
    attr : dict
        Player attribute pointer configurations.

    Raises
    ------
    NotImplementedError
        If running on non-Windows platform.
    """

    def __init__(self, pc: pymem.Pymem, game_module: int):
        self.os = platform.system()
        self.pc = pc
        self.game_module = game_module
        self.attr = PLAYER_PTR

        if self.os != "Windows":
            raise NotImplementedError("Only Windows is supported")

    def is_gem_high(self) -> bool:
        """Check if gem count is high (>=100).

        Returns
        -------
        bool
            True if gem count is 100 or higher, False otherwise.
        """
        value = self.get_value("gemHigh")
        if value is None:
            logger.debug("Failed to read gemHigh value")
            return False
        return value >= 100

    def get_ptr_addr(self, base: int, offsets: list[int]) -> int:
        """Resolve pointer chain to get final address.

        Parameters
        ----------
        base : int
            Base memory address.
        offsets : list[int]
            Chain of offsets to follow.

        Returns
        -------
        int
            Final resolved memory address.

        Raises
        ------
        pymem.exception.MemoryReadError
            If memory read fails at any point in the chain.
        """
        try:
            addr = int(self.pc.read_int(base))
            for offset in offsets[:-1]:
                addr = int(self.pc.read_int(addr + offset))
            return addr + offsets[-1]
        except pymem.exception.MemoryReadError as e:
            raise e

    def get_type(self, attr_type: str, address: int) -> float | None:
        """Read a typed value from memory.

        Parameters
        ----------
        attr_type : str
            Type name (e.g., 'float', 'double', 'int').
        address : int
            Memory address to read from.

        Returns
        -------
        float | None
            Value read from memory, or None if type is unknown.

        Raises
        ------
        pymem.exception.MemoryReadError
            If memory read fails.
        """
        try:
            return getattr(self.pc, f"read_{attr_type}")(address)  # type: ignore[no-any-return]
        except AttributeError:
            logger.error(f"Unknown type: {attr_type}")
            return None
        except pymem.exception.MemoryReadError as e:
            raise e

    def get_value(self, attribute: str) -> float | None:
        """Get player attribute value from memory.

        Parameters
        ----------
        attribute : str
            Attribute name (e.g., 'hp', 'gems', 'xpos').

        Returns
        -------
        float | None
            Attribute value, or None if read fails.
        """
        if attribute not in self.attr:
            logger.error(f"Unknown attribute: {attribute}")
            return None

        attr_data = cast(dict[str, Any], self.attr[attribute])
        if "bases" in attr_data:
            bases: list[int] = cast(list[int], attr_data["bases"])
            offsets_list: list[list[int]] = cast(list[list[int]], attr_data["offsets"])
        else:
            bases = [cast(int, attr_data["base"])]
            offsets_list = [cast(list[int], attr_data["offsets"])]

        for base, offsets in zip(bases, offsets_list, strict=False):
            try:
                address = self.get_ptr_addr(self.game_module + base, offsets)
                value = self.get_type(cast(str, attr_data["type"]), address)
                return value
            except pymem.exception.MemoryReadError:
                continue

        logger.debug(f"Failed to read {attribute} from all available addresses")
        return None

    def validate_connection(self) -> bool:
        """Validate that connection to game process is still active.

        Returns
        -------
        bool
            True if connection is valid, False otherwise.
        """
        try:
            # Attempt to read a known value to check connection
            self.get_value("hp")
            return True
        except Exception as e:
            logger.error(f"Lost connection to game process: {e}")
            return False
