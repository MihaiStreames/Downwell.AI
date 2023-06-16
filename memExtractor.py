from pymem.process import *
import time

class Player:
    def __init__(self, pc, gameModule):
        self.pc = pc
        self.gameModule = gameModule
        self.attributes = {
            "ypos": {"base": 0x0005220C, "offsets": [0x360], "type": "float"},
            "xpos": {"base": 0x00534288, "offsets": [0x340, 0x2F4, 0x18C, 0x60, 0x20, 0x14C, 0xB0], "type": "float"},
            "hp": {"base": 0x004A5E50, "offsets": [0x708, 0xC, 0x24, 0x10, 0x9C0, 0x390], "type": "double"},
            "gems": {"base": 0x00757BF0, "offsets": [0x24, 0x10, 0x330, 0xE0, 0x50, 0x9A8, 0x350], "type": "double"},
            "ammo": {"bases": [0x00757C80, 0x00757BF8, 0x00757978], "offsets": [[0x88, 0x160, 0x50, 0x804, 0x150], [0x24, 0x10, 0x4D4, 0x160, 0x50, 0xE4C, 0x660], [0x324, 0xE0, 0x8, 0x8, 0x50, 0xF78, 0xA0]], "type": "double"},
            "gemHigh": {"base": 0x004A5E50, "offsets": [0x708, 0xC, 0x24, 0x10, 0x9E4, 0x480], "type": "double"},
            "combo": {"base": 0x00757C80, "offsets": [0x168, 0x160, 0x8, 0x8, 0x50, 0x108, 0x2C0], "type": "double"}
        }

    def getPointerAddress(self, base, offsets):
        addr = self.pc.read_int(base)
        for i in offsets[:-1]:
            addr = self.pc.read_int(addr + i)

        return addr + offsets[-1]

    def getType(self, type, address):
        if type == 'double':
            return self.pc.read_double(address)
        elif type == 'float':
            return self.pc.read_float(address)

    def getValue(self, attribute):
        if attribute == 'ammo':
            for i in range(len(self.attributes[attribute]["bases"])):
                try:
                    return self.getType(self.attributes[attribute]["type"], self.getPointerAddress(self.gameModule + self.attributes[attribute]["bases"][i], self.attributes[attribute]["offsets"][i])), i
                except pymem.exception.MemoryReadError:
                    continue
            print("Error: Ammo not found")
        else:
            return self.getType(self.attributes[attribute]["type"], self.getPointerAddress(self.gameModule + self.attributes[attribute]["base"], self.attributes[attribute]["offsets"]))

    def isGemHigh(self):
        return self.getValue('gemHigh') >= 100

if __name__ == "__main__":
    pc = pymem.Pymem("downwell.exe")
    gameModule = module_from_name(pc.process_handle, "downwell.exe").lpBaseOfDll
    player = Player(pc, gameModule)

    while True:
        try:
            print("Y: " + str(player.getValue('ypos')) + " | X: " + str(player.getValue('xpos')))
            print("HP: " + str(player.getValue('hp')) + " | ", end="")
            print("Gems: " + str(player.getValue('gems')) + " | ", end="")
            print("Ammo: " + str(player.getValue('ammo')) + " | ", end="")
            print("Gem Combo: " + str(player.getValue('gemHigh')) + " / Gem High: " + str(player.isGemHigh()) + " | ", end="")
            print("Current Combo: " + str(player.getValue('combo')))

            time.sleep(1)

        except pymem.exception.MemoryReadError as e:
            print("Error: " + str(e))
            time.sleep(5)
            continue