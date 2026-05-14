import importlib


def test_main_importable() -> None:
    importlib.import_module("src.__main__")
