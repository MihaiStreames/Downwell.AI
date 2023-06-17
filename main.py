from memExtractor import Player
from customEnv import CustomDownwellEnvironment
from agent import DQNAgent

from pymem.process import *

import time


def main():
    proc = pymem.Pymem("downwell.exe")
    gameModule = module_from_name(proc.process_handle, "downwell.exe").lpBaseOfDll
    player = Player(proc, gameModule)

    gameEnv = CustomDownwellEnvironment()
    gameEnv.reset(player)
    time.sleep(2)

    # TODO: Agent initialization

    while True:
        done = gameEnv.isGameOver(player)
        if not done:
            print(f'X: {player.getValue("xpos")} | Y: {player.getValue("ypos")}')
            print(f'Health: {player.getValue("hp")}', end="")
            print(f' | Ammo: {player.getValue("ammo")[0]}', end="")
            print(f' | Gems: {player.getValue("gems")} / Gem High: {player.isGemHigh()}', end="")
            print(f' | Gem Combo: {player.getValue("gemHigh")} / Current Combo: {player.getValue("combo")}')
            time.sleep(1)

        # TODO: Agent training

        if done:
            gameEnv.reset(player)
            print("Game Over :(")
            break


if __name__ == "__main__":
    main()