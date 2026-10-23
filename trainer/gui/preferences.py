"""Machine-local GUI preferences, separate from portable training TOMLs."""
import json
import os
from pathlib import Path
import tempfile

from PySide6 import QtCore


def preferences_path():
    base = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.GenericConfigLocation)
    return Path(base) / "mage-flow-trainer" / "gui.json"


class Preferences:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else preferences_path()
        self.values = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.values = data
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            print(f"Could not read GUI preferences {self.path}: {exc}; using defaults.")

    def set(self, key, value):
        values = {**self.values, key: value}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix="gui-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(values, stream, indent=2)
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self.values = values
