import platform
from pymem.process import *
from src.game_attributes import *


class Player:
    def __init__(self, pc, gameModule):
        self.os = platform.system()
        self.pc = pc
        self.gameModule = gameModule

        self.attr = PLAYER_PTR_WIN if self.os == "Windows" else PLAYER_PTR_LINUX

    def is_gem_high(self):
        return self.get_value('gemHigh') >= 100

    def get_ptr_addr(self, base, offsets):
        if self.os == "Windows":
            addr = self.pc.read_int(base)
            for offset in offsets[:-1]:
                addr = self.pc.read_int(addr + offset)
            return addr + offsets[-1]
        elif self.os == "Linux":
            # Linux-specific logic here
            pass
        else:
            raise NotImplementedError("Unsupported OS")

    def get_ptr_addr_windows(self, base, offsets):
        addr = self.pc.read_int(base)
        for i in offsets[:-1]:
            addr = self.pc.read_int(addr + i)
        return addr + offsets[-1]

    def get_type(self, attr_type, address):
        return getattr(self.pc, f'read_{attr_type}')(address)

    def get_value(self, attribute):
        attr_data = self.attr[attribute]
        bases = attr_data.get("bases", [attr_data["base"]])
        offsets_list = attr_data.get("offsets", [attr_data["offsets"]])

        for base, offsets in zip(bases, offsets_list):
            try:
                address = self.get_ptr_addr(self.gameModule + base, offsets)
                return self.get_type(attr_data["type"], address)
            except pymem.exception.MemoryReadError:
                continue
        print(f"Error: {attribute} not found")