"""Optional Git update check before importing the training GUI."""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Update:
    branch: str
    upstream: str
    head: str
    ahead: int
    behind: int
    dirty: bool


def git(root, *args, timeout=15):
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=8")
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=True,
    )
    return result.stdout.strip()


def check_update(root):
    """Fetch only the current branch's remote; no checkout or working-tree edits."""
    branch = git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    upstream = git(root, "rev-parse", "--abbrev-ref", "@{upstream}")
    remote = git(root, "config", "--get", f"branch.{branch}.remote")
    git(root, "fetch", "--quiet", "--no-tags", remote)
    ahead, behind = map(
        int,
        git(root, "rev-list", "--left-right", "--count", "HEAD...@{upstream}").split(),
    )
    return Update(
        branch,
        upstream,
        git(root, "rev-parse", "HEAD"),
        ahead,
        behind,
        bool(git(root, "status", "--porcelain", "--untracked-files=no")),
    )


def apply_update(root, expected):
    # Recheck after the dialog: another process may have changed the checkout.
    if git(root, "symbolic-ref", "--quiet", "--short", "HEAD") != expected.branch:
        raise RuntimeError("The selected branch changed; update manually and restart.")
    if git(root, "rev-parse", "--abbrev-ref", "@{upstream}") != expected.upstream:
        raise RuntimeError("The upstream changed; update manually and restart.")
    if git(root, "rev-parse", "HEAD") != expected.head:
        raise RuntimeError("The local commit changed; update manually and restart.")
    if expected.ahead or git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(
            "Local commits or tracked edits need manual review before updating."
        )
    return git(root, "pull", "--ff-only", "--no-rebase", "--no-autostash", timeout=120)


def _wait(app, label, operation):
    """Keep the small startup dialog responsive during network operations."""
    from PySide6 import QtCore, QtWidgets

    progress = QtWidgets.QProgressDialog(label, "", 0, 0)
    progress.setWindowTitle("Mage-Flow Trainer")
    progress.setCancelButton(None)
    progress.setMinimumDuration(0)
    progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    progress.show()
    app.processEvents()
    loop = QtCore.QEventLoop()
    timer = QtCore.QTimer()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(operation)
        timer.timeout.connect(lambda: loop.quit() if future.done() else None)
        timer.start(50)
        loop.exec()
        timer.stop()
        progress.close()
        return future.result()


def offer_update(app, root):
    """Return True only after an approved successful update, requesting restart."""
    from PySide6 import QtWidgets

    try:
        update = _wait(app, "Checking for trainer updates…", lambda: check_update(root))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"Update check skipped: {exc}", flush=True)
        return False
    if not update.behind:
        return False
    if update.ahead or update.dirty:
        QtWidgets.QMessageBox.information(
            None,
            "Trainer update available",
            f"{update.upstream} has {update.behind} new commit(s), but this checkout has "
            "local commits or tracked edits. Update manually when ready. "
            "The GUI will start with your current version.",
        )
        return False
    answer = QtWidgets.QMessageBox.question(
        None,
        "Trainer update available",
        f"{update.branch} is {update.behind} commit(s) behind {update.upstream}.\n\n"
        "Update with git pull --ff-only before starting?\n"
        "Changed dependency files will also refresh packages in this GUI's venv before restarting.\n"
        "Choose No to start the current version.",
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.No,
        QtWidgets.QMessageBox.StandardButton.No,
    )
    if answer != QtWidgets.QMessageBox.StandardButton.Yes:
        return False
    try:
        _wait(app, "Updating trainer…", lambda: apply_update(root, update))
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        QtWidgets.QMessageBox.warning(
            None,
            "Trainer update failed",
            f"{detail}\n\nThe GUI will start with the current checkout. No reset or stash was performed.",
        )
        return False
    return True


def offer_dependency_update(root):
    """Catch manual pulls and existing installs without a successful-sync marker."""
    from PySide6 import QtWidgets
    from ..tools.update_dependencies import dependencies_current, in_venv

    if not in_venv() or dependencies_current(root):
        return False
    answer = QtWidgets.QMessageBox.question(
        None, "Refresh trainer packages",
        "Dependency files have changed, or this venv has not recorded a successful package refresh.\n\n"
        "Install the repository's pinned packages into the active venv before starting? "
        "This includes the required LyCORIS revision.\n"
        "Choose No to continue with the existing packages.",
        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        QtWidgets.QMessageBox.StandardButton.Yes,
    )
    return answer == QtWidgets.QMessageBox.StandardButton.Yes


def restart(*, refresh_dependencies):
    from .launcher import REFRESH_DEPENDENCIES, RESTART_GUI

    # Let Python/Qt shut down normally. The stdlib-only parent waits for this
    # process to exit before installing packages or launching another GUI.
    print("Closing startup GUI for update...", flush=True)
    raise SystemExit(REFRESH_DEPENDENCIES if refresh_dependencies else RESTART_GUI)


def main():
    import sys

    from PySide6 import QtWidgets

    skip = "--no-update-check" in sys.argv
    skip_dependencies = "--no-dependency-check" in sys.argv
    sys.argv[:] = [arg for arg in sys.argv if arg not in ("--no-update-check", "--no-dependency-check")]
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Mage-Flow Trainer")
    root = Path(__file__).resolve().parents[2]
    updated = not skip and offer_update(app, root)
    from ..tools.update_dependencies import in_venv
    if updated:
        restart(refresh_dependencies=not skip_dependencies and in_venv())
    if not skip_dependencies and offer_dependency_update(root):
        restart(refresh_dependencies=True)
    from .app import main as launch

    launch()
