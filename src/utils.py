import sys
from pathlib import Path

def get_base_path():
    """
    Returns base path for:
    - normal python execution
    - PyInstaller frozen app
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent

def resource_path(relative_path: str) -> Path:
    return get_base_path() / relative_path
