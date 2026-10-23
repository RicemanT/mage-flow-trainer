"""Dependency refresh uses the active venv and only records successful installs."""
import json
from pathlib import Path

import pytest

from trainer.tools import update_dependencies as updater


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root, venv = tmp_path / "fork", tmp_path / "custom env"
    root.mkdir()
    venv.mkdir()
    (root / "requirements.txt").write_text("lycoris-lora==4.1.3\n")
    (root / "pyproject.toml").write_text('[project]\nname="fork"\n')
    monkeypatch.setattr(updater.sys, "prefix", str(venv))
    monkeypatch.setattr(updater.sys, "base_prefix", str(tmp_path / "python"))
    monkeypatch.setattr(updater.sys, "executable", str(venv / "python.exe"))
    monkeypatch.setattr(updater.shutil, "which", lambda name: None)
    monkeypatch.setattr(updater.importlib.util, "find_spec", lambda name: True)
    return root, venv


def test_content_fingerprint_and_success_marker(environment):
    root, venv = environment
    assert not updater.dependencies_current(root)
    updater.record_current(root)
    assert updater.dependencies_current(root)
    (root / "requirements.txt").touch()
    assert updater.dependencies_current(root)
    (root / "requirements.txt").write_text("lycoris-lora==4.1.4\n")
    assert not updater.dependencies_current(root)
    updater.record_current(root)
    (root / "uv.lock").write_text("changed lock")
    assert not updater.dependencies_current(root)
    (venv / ".mageflow-trainer-dependencies.json").write_text("broken")
    assert not updater.dependencies_current(root)


def test_pip_targets_exact_interpreter_and_pins(environment):
    root, venv = environment
    command, _ = updater.install_command(root)
    assert command[:4] == [str(venv / "python.exe"), "-m", "pip", "install"]
    assert command[command.index("-r") + 1] == str(root / "requirements.txt")
    assert "--upgrade" in command
    assert updater.TORCH_INDEX in command


def test_uv_targets_active_venv_and_preserves_optional_packages(environment, monkeypatch):
    root, venv = environment
    (root / "uv.lock").write_text("locked")
    monkeypatch.setattr(updater.shutil, "which", lambda name: "/bin/uv")
    command, env = updater.install_command(root)
    assert command[:2] == ["/bin/uv", "sync"]
    assert env["UV_PROJECT_ENVIRONMENT"] == str(venv)
    assert "--locked" in command and "--inexact" in command
    assert command[command.index("--python") + 1] == str(venv / "python.exe")


def test_failure_retries_but_success_skips_repeat_install(environment, monkeypatch):
    root, _ = environment
    calls = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError("offline")

    monkeypatch.setattr(updater, "_run_installer", fail)
    with pytest.raises(RuntimeError, match="offline"):
        updater.refresh(root)
    assert not updater.dependencies_current(root)
    monkeypatch.setattr(updater, "_run_installer", lambda *args: calls.append(args))
    assert updater.refresh(root)
    assert not updater.refresh(root)
    assert len(calls) == 2


def test_mid_install_source_change_does_not_record_success(environment, monkeypatch):
    root, _ = environment
    def install(*args):
        (root / "requirements.txt").write_text("changed during install")
    monkeypatch.setattr(updater, "_run_installer", install)
    with pytest.raises(RuntimeError, match="changed during"):
        updater.refresh(root)
    assert not updater.dependencies_current(root)


def test_global_python_not_modified(environment, monkeypatch):
    root, _ = environment
    monkeypatch.setattr(updater, "in_venv", lambda: False)
    with pytest.raises(RuntimeError, match="global Python"):
        updater.refresh(root)


def test_missing_pip_bootstrapped_in_active_venv(environment, monkeypatch):
    root, venv = environment
    commands = []
    monkeypatch.setattr(updater.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(updater, "_run_installer", lambda cmd, *args: commands.append(cmd))
    assert updater.refresh(root)
    assert commands[0] == [str(venv / "python.exe"), "-m", "ensurepip", "--upgrade"]
    assert commands[1][1:4] == ["-m", "pip", "install"]


def test_failed_worker_never_launches_gui(environment, monkeypatch):
    root, _ = environment
    monkeypatch.setattr(updater, "refresh", lambda root: (_ for _ in ()).throw(RuntimeError("broken")))
    monkeypatch.setattr(updater.os, "execv", lambda *args: pytest.fail("GUI started after failed update"))
    assert updater.main(["--root", str(root), "--restart-gui"]) == 1


def test_worker_restarts_gui_only_after_install(environment, monkeypatch):
    root, _ = environment
    calls = []
    monkeypatch.setattr(updater, "_run_installer", lambda *args: calls.append("installed"))
    monkeypatch.setattr(updater.os, "chdir", lambda path: calls.append(path))
    monkeypatch.setattr(updater.os, "execv", lambda exe, argv: calls.append(argv))
    assert updater.main(["--root", str(root), "--restart-gui"]) == 0
    assert calls[0] == "installed"
    assert updater.dependencies_current(root)
    assert calls[-1][1:] == ["-m", "trainer.gui", "--no-update-check"]
