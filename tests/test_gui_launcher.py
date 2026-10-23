"""Restart sequencing and real child-process shutdown, without package installs."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from trainer.gui import launcher


@pytest.mark.parametrize("codes,expected", [
    ([0], 0), ([9], 9), ([75, 0, 0], 0), ([75, 3], 3),
    ([76, 0], 0), ([76, 76, 76], 1),
])
def test_supervisor_sequence(monkeypatch, tmp_path, codes, expected):
    root = tmp_path / "trainer with spaces"
    calls = []
    remaining = iter(codes)
    def run(argv, *, cwd):
        assert cwd == root
        calls.append(argv)
        return subprocess.CompletedProcess(argv, next(remaining))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.supervise(root, ["--no-dependency-check", "argument with spaces"]) == expected
    assert len(calls) == len(codes)
    gui_calls = [c for c in calls if launcher.CHILD_FLAG in c]
    for call in gui_calls:
        assert call[0] == sys.executable
        assert "--no-dependency-check" in call
        assert "argument with spaces" in call
    for call in gui_calls[1:]:
        assert call.count("--no-update-check") == 1
    if codes[0] == 75:
        assert calls[1] == [sys.executable, str(root / "trainer/tools/update_dependencies.py"),
                            "--root", str(root)]


def test_supervisor_import_does_not_load_qt_or_torch():
    subprocess.run([sys.executable, "-c",
        "import sys; import trainer.gui.launcher; "
        "assert not any(n.startswith(('PySide6', 'torch')) for n in sys.modules)"], check=True)


@pytest.mark.parametrize("installer_code", [0, 4])
def test_real_qt_shutdown_before_installer(tmp_path, installer_code):
    root = tmp_path / "checkout with spaces"
    gui = root / "trainer/gui"
    worker = root / "trainer/tools"
    gui.mkdir(parents=True)
    worker.mkdir()
    (root / "trainer/__init__.py").touch()
    (gui / "__init__.py").touch()
    # Use the actual launcher, with a tiny GUI replacing the full application.
    (gui / "launcher.py").write_text(Path(launcher.__file__).read_text())
    (gui / "__main__.py").write_text(
        "from .launcher import main\nraise SystemExit(main())\n")
    (gui / "startup.py").write_text('''
import atexit
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    assert "argument with spaces" in sys.argv
    assert "--gui-child" not in sys.argv
    if Path("installed").exists():
        assert "--no-update-check" in sys.argv
        Path("reopened").touch()
        return
    atexit.register(lambda: Path("gui exited").touch())
    raise SystemExit(75)
''')
    (worker / "update_dependencies.py").write_text(f'''
from pathlib import Path
import sys
assert Path("gui exited").exists(), "Installer started before GUI shutdown"
assert not any(n.startswith(("PySide6", "torch")) for n in sys.modules)
assert sys.argv[1:] == ["--root", str(Path.cwd())]
Path("installed").touch()
raise SystemExit({installer_code})
''')
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [sys.executable, "-m", "trainer.gui", "argument with spaces"],
        cwd=root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == installer_code, result.stdout + result.stderr
    assert (root / "gui exited").exists()
    assert (root / "installed").exists()
    assert (root / "reopened").exists() == (installer_code == 0)
