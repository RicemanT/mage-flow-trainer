import os
import subprocess

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from trainer.gui import startup


def run(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def repos(tmp_path):
    remote, author, local = [
        tmp_path / name for name in ("fork.git", "author", "local")
    ]
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True
    )
    author.mkdir()
    run(author, "init", "-b", "feature")
    run(author, "config", "user.email", "test@example.invalid")
    run(author, "config", "user.name", "Test")
    (author / "file.txt").write_text("first")
    run(author, "add", ".")
    run(author, "commit", "-m", "initial")
    run(author, "remote", "add", "my-fork", str(remote))
    run(author, "push", "-u", "my-fork", "feature")
    subprocess.run(
        ["git", "clone", "-b", "feature", "-o", "my-fork", str(remote), str(local)],
        check=True,
        capture_output=True,
    )
    return author, local


def advance(author):
    (author / "file.txt").write_text("second")
    run(author, "commit", "-am", "update")
    run(author, "push")


def test_fork_upstream_fetch_and_fast_forward(repos):
    author, local = repos
    assert startup.check_update(local).behind == 0
    advance(author)
    update = startup.check_update(local)
    assert update.upstream == "my-fork/feature"
    assert update.behind == 1 and update.ahead == 0
    assert (local / "file.txt").read_text() == "first"
    (local / "personal.toml").write_text("local config")
    startup.apply_update(local, update)
    assert (local / "file.txt").read_text() == "second"
    assert (local / "personal.toml").read_text() == "local config"
    assert startup.check_update(local).behind == 0


def test_dirty_checkout_is_preserved(repos):
    author, local = repos
    advance(author)
    (local / "file.txt").write_text("my changes")
    update = startup.check_update(local)
    assert update.dirty
    with pytest.raises(RuntimeError, match="tracked edits"):
        startup.apply_update(local, update)
    assert (local / "file.txt").read_text() == "my changes"


def test_divergence_is_not_merged(repos):
    author, local = repos
    run(local, "config", "user.email", "test@example.invalid")
    run(local, "config", "user.name", "Test")
    (local / "mine.txt").write_text("mine")
    run(local, "add", ".")
    run(local, "commit", "-m", "local")
    advance(author)
    update = startup.check_update(local)
    assert update.ahead == update.behind == 1
    with pytest.raises(RuntimeError, match="Local commits"):
        startup.apply_update(local, update)


@pytest.mark.parametrize("answer, expected", [("No", False), ("Yes", True)])
def test_update_requires_consent(monkeypatch, answer, expected):
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(startup, "_wait", lambda app, label, op: op())
    monkeypatch.setattr(
        startup,
        "check_update",
        lambda root: startup.Update("main", "fork/main", "abc", 0, 1, False),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: getattr(QtWidgets.QMessageBox.StandardButton, answer),
    )
    applied = []
    monkeypatch.setattr(startup, "apply_update", lambda *args: applied.append(True))
    assert startup.offer_update(app, "/repo") is expected
    assert bool(applied) is expected


def test_missing_git_or_offline_still_starts(monkeypatch):
    monkeypatch.setattr(startup, "_wait", lambda app, label, op: op())
    for error in (
        FileNotFoundError("git"),
        subprocess.TimeoutExpired("git", 15),
        subprocess.CalledProcessError(1, "git"),
    ):

        def fail(root, error=error):
            raise error

        monkeypatch.setattr(startup, "check_update", fail)
        assert not startup.offer_update(None, "/repo")


def test_detached_or_no_upstream_skips_update(repos, monkeypatch):
    _, local = repos
    monkeypatch.setattr(startup, "_wait", lambda app, label, op: op())
    run(local, "branch", "--unset-upstream")
    assert not startup.offer_update(None, local)
    run(local, "checkout", "--detach")
    assert not startup.offer_update(None, local)


def test_untracked_collision_is_not_overwritten(repos):
    author, local = repos
    (author / "personal.toml").write_text("upstream")
    run(author, "add", ".")
    run(author, "commit", "-m", "add file")
    run(author, "push")
    (local / "personal.toml").write_text("mine")
    update = startup.check_update(local)
    with pytest.raises(subprocess.CalledProcessError):
        startup.apply_update(local, update)
    assert (local / "personal.toml").read_text() == "mine"
    assert run(local, "rev-parse", "HEAD") == update.head
