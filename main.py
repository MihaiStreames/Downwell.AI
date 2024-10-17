from src.mem_extractor import Player
from src.custom_env import CustomDownwellEnvironment

from pymem.process import *
import platform
import time


def get_game_module(proc, executable_name):
    try:
        return module_from_name(proc.process_handle, executable_name).lpBaseOfDll
    except Exception as e:
        print(f"Error finding module '{executable_name}': {str(e)}")
        raise


def main():
    os_type = platform.system()

    # Define the executable name based on OS
    executable_name = "downwell.exe" if os_type == "Windows" else "downwell_linux_executable"

    try:
        proc = pymem.Pymem(executable_name)
    except Exception as e:
        print(f"Error opening process '{executable_name}': {str(e)}")
        return

    try:
        gameModule = get_game_module(proc, executable_name)
    except Exception:
        return

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