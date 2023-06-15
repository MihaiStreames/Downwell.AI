from pymem.process import *
import time

pc = pymem.Pymem("downwell.exe")
gameModule = module_from_name(pc.process_handle, "downwell.exe").lpBaseOfDll

def GetPointerAddress(base, offsets):
    addr = pc.read_int(base)
    for i in offsets[:-1]:
        addr = pc.read_int(addr + i)

    return addr + offsets[-1]

class HP:
    def __init__(self):
        self.__base = 0x004A5E50
        self.__offsets = [0x708, 0xC, 0x24, 0x10, 0x9C0, 0x390]

    def getValue(self):
        return pc.read_double(GetPointerAddress(gameModule + self.__base, self.__offsets))

class Gems:
    def __init__(self):
        self.__base = 0x00757BF0
        self.__offsets = [0x24, 0x10, 0x330, 0xE0, 0x50, 0x9A8, 0x350]

    def getValue(self):
        return pc.read_double(GetPointerAddress(gameModule + self.__base, self.__offsets))

class Ammo:
    def __init__(self):
        self.__base = 0x00536180
        self.__offsets = [0x10, 0x1F0, 0x78, 0x24, 0x10, 0xF24, 0x3C0]

    def getValue(self):
        return pc.read_double(GetPointerAddress(gameModule + self.__base, self.__offsets))

class GemHigh:
    def __init__(self):
        self.__base = 0x004A5E50
        self.__offsets = [0x708, 0xC, 0x24, 0x10, 0x9E4, 0x480]
        self.__isGemHigh = False

    def getValue(self):
        value = pc.read_double(GetPointerAddress(gameModule + self.__base, self.__offsets))
        self.__isGemHigh = value >= 100
        return value

    def isGemHigh(self):
        return self.__isGemHigh

class Combo:
    def __init__(self):
        self.__base = 0x00757C80
        self.__offsets = [0x168, 0x160, 0x8, 0x8, 0x50, 0x108, 0x2C0]

    def getValue(self):
        return pc.read_double(GetPointerAddress(gameModule + self.__base, self.__offsets))

if __name__ == "__main__":
    hp = HP()
    gems = Gems()
    ammo = Ammo()
    gemhigh = GemHigh()
    combo = Combo()

    while True:
        try:
            print("HP: " + str(hp.getValue()) +
                  " | Gems: " + str(gems.getValue()) +
                  " | Ammo: " + str(ammo.getValue()) +
                  " | Gem Combo: " + str(gemhigh.getValue()) +
                  " | Gem High: " + str(gemhigh.isGemHigh()) +
                  " | Current Combo: " + str(combo.getValue()))
            time.sleep(1)
        except:
            print("End of Level")
            time.sleep(1)