import platform

from pymem.process import *

from utils.game_attributes import *


class Player:
    def __init__(self, pc, game_module):
        self.os = platform.system()
        self.pc = pc
        self.game_module = game_module
        self.attr = PLAYER_PTR

        if self.os != "Windows": raise NotImplementedError("Currently only Windows is supported")

    def is_gem_high(self) -> bool:
        if self.get_value('gemHigh') is None:
            print("Failed to read gemHigh value")
            return False
        return self.get_value('gemHigh') >= 100

    def get_ptr_addr(self, base: int, offsets: list) -> int:
        try:
            addr = self.pc.read_int(base)
            for offset in offsets[:-1]:
                addr = self.pc.read_int(addr + offset)
            return addr + offsets[-1]
        except pymem.exception.MemoryReadError as e:
            raise e

    def get_type(self, attr_type: str, address: int):
        try:
            return getattr(self.pc, f'read_{attr_type}')(address)
        except AttributeError:
            print(f"Unknown type: {attr_type}")
            return None
        except pymem.exception.MemoryReadError as e:
            raise e

    def get_value(self, attribute: str):
        if attribute not in self.attr:
            print(f"Unknown attribute: {attribute}")
            return None

        attr_data = self.attr[attribute]
        if "bases" in attr_data:
            bases = attr_data["bases"]
            offsets_list = attr_data["offsets"]
        else:
            bases = [attr_data["base"]]
            offsets_list = [attr_data["offsets"]]

        for base, offsets in zip(bases, offsets_list):
            try:
                address = self.get_ptr_addr(self.game_module + base, offsets)
                value = self.get_type(attr_data["type"], address)
                return value
            except pymem.exception.MemoryReadError:
                continue

        print(f"Failed to read {attribute} from all available addresses")
        return None

    def validate_connection(self) -> bool:
        try:
            # Attempt to read a known value to check connection
            self.get_value('hp')
            return True
        except Exception as e:
            print(f"Lost connection to game process: {e}")
            return False
