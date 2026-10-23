"""Supervise GUI restarts without importing Qt or locking its DLLs on Windows."""
from pathlib import Path
import subprocess
import sys


REFRESH_DEPENDENCIES = 75
RESTART_GUI = 76
CHILD_FLAG = "--gui-child"


def supervise(root, gui_args):
    args = list(gui_args)
    # A successful refresh must not request another installation indefinitely.
    for attempt in range(3):
        result = subprocess.run(
            [sys.executable, "-m", "trainer.gui", CHILD_FLAG, *args], cwd=root,
        )
        if result.returncode not in (REFRESH_DEPENDENCIES, RESTART_GUI):
            return result.returncode
        if attempt == 2:
            print("GUI keeps requesting a restart. Run the installer manually.",
                  file=sys.stderr, flush=True)
            return 1
        # subprocess.run has waited for Qt to terminate, including DLL cleanup.
        if result.returncode == REFRESH_DEPENDENCIES:
            print("GUI closed. Checking and refreshing trainer packages...", flush=True)
            update = subprocess.run(
                [sys.executable, str(root / "trainer/tools/update_dependencies.py"),
                 "--root", str(root)], cwd=root,
            )
            if update.returncode:
                print(f"Package updater exited with code {update.returncode}; GUI not restarted.",
                      file=sys.stderr, flush=True)
                return update.returncode
        if "--no-update-check" not in args:
            args.append("--no-update-check")
        print("Starting updated GUI...", flush=True)
    return 1


def main():
    if sys.argv[1:2] == [CHILD_FLAG]:
        del sys.argv[1]
        from .startup import main as start_gui
        start_gui()
        return 0
    try:
        return supervise(Path(__file__).resolve().parents[2], sys.argv[1:])
    except OSError as exc:
        print(f"Could not start trainer process: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        return 130
