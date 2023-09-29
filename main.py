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
        done = gameEnv.is_game_over(player)
        if not done:
            print(f'X: {player.get_value("xpos")} | Y: {player.get_value("ypos")}')
            print(f'Health: {player.get_value("hp")}', end="")
            print(f' | Ammo: {player.get_value("ammo")[0]}', end="")
            print(f' | Gems: {player.get_value("gems")} / Gem High: {player.is_gem_high()}', end="")
            print(f' | Gem Combo: {player.get_value("gemHigh")} / Current Combo: {player.get_value("combo")}')
            time.sleep(1)

        # TODO: Agent training

        if done:
            gameEnv.reset(player)
            print("Game Over :(")
            break


if __name__ == "__main__":
    main()