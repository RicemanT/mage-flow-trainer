import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from trainer.gui import app as gui_module, bridge
from trainer.gui.preferences import Preferences


@pytest.fixture
def gui(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(gui_module, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(gui_module, "detect_gpus", lambda: [(0, "Test GPU")])
    window = gui_module.TrainingGUI(preferences=Preferences(tmp_path / "settings/gui.json"))
    window._apply(bridge.defaults() | {"dataset.path": "/images", "train.run_name": "example"})
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def test_defaults_and_no_save_while_editing(gui):
    assert gui.autosave_chk.isChecked()
    assert gui.config_dir == gui_module.CONFIG_DIR
    gui.editors["train.run_name"].set("edited")
    gui._on_edit()
    assert not list(gui.config_dir.glob("*.toml"))
    assert not gui.preferences.path.exists()


def test_invalid_preferences_use_defaults(tmp_path):
    path = tmp_path / "gui.json"
    for content in ("broken", "null", "[]"):
        path.write_text(content)
        assert Preferences(path).values == {}


def test_picker_mtime_order_with_external_file_and_preserved_selection(gui, tmp_path):
    gui.config_dir.mkdir()
    paths = [gui.config_dir / "a-old.toml", gui.config_dir / "z-new.toml", tmp_path / "external.toml"]
    for i, path in enumerate(paths):
        bridge.write_toml(path, gui.collect() | {"train.run_name": path.stem})
        os.utime(path, ns=(1000 + i, 1000 + i))
    gui._external.extend([paths[2], paths[2], tmp_path / "gone.toml"])
    gui._load_presets(select=paths[0])
    assert [Path(gui.preset_combo.itemData(i)) for i in range(3)] == list(reversed(paths))
    assert gui.collect()["train.run_name"] == "a-old"
    gui.editors["train.run_name"].set("unsaved")
    gui._load_presets(select=paths[0], apply=False)
    assert gui.collect()["train.run_name"] == "unsaved"
    assert gui.preset_combo.currentData() == str(paths[0])
    gui._load_presets(apply=False)
    assert gui.preset_combo.currentIndex() == -1
    assert gui.collect()["train.run_name"] == "unsaved"


def test_start_autosaves_but_off_uses_current_form_without_touching_original(gui):
    first = gui._training_config(gui.collect())
    original = first.read_bytes()
    gui.autosave_chk.setChecked(False)
    assert Preferences(gui.preferences.path).values["autosave"] is False
    gui.editors["optimizer.lr"].set(0.003)
    temporary = gui._training_config(gui.collect())
    assert temporary.parent != gui.config_dir
    assert bridge.read_toml(temporary)["optimizer"]["lr"] == 0.003
    assert first.read_bytes() == original
    assert gui._current_path == first
    # Explicit Save remains available with autosave disabled.
    gui._save()
    assert bridge.read_toml(first)["optimizer"]["lr"] == 0.003


def test_custom_folder_persisted_and_external_source_untouched(gui, tmp_path):
    source = tmp_path / "source.toml"
    source.write_text('# keep my comments\n[train]\nrun_name="example"\n')
    gui._current_path = source
    custom = tmp_path / "my TOMLs"
    assert gui._set_config_dir(str(custom))
    assert Preferences(gui.preferences.path).values["config_dir"] == str(custom)
    target = gui._training_config(gui.collect())
    assert target == custom / "example.toml"
    assert source.read_text().startswith("# keep my comments")
    assert gui._set_config_dir("")
    assert gui.config_dir == gui_module.CONFIG_DIR
    assert Preferences(gui.preferences.path).values["config_dir"] == ""


def test_save_collision_still_requires_confirmation(gui, monkeypatch):
    gui.config_dir.mkdir()
    target = gui._managed_target(gui.collect())
    target.write_text("# other config")
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a: QtWidgets.QMessageBox.StandardButton.No)
    assert gui._training_config(gui.collect()) is None
    assert target.read_text() == "# other config"


def test_invalid_folder_keeps_previous_preference(gui, tmp_path, monkeypatch):
    assert gui._set_config_dir(str(tmp_path / "valid"))
    old = gui.config_dir
    invalid = tmp_path / "not-a-folder"
    invalid.write_text("file")
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a: None)
    assert not gui._set_config_dir(str(invalid))
    assert gui.config_dir == old
    assert Preferences(gui.preferences.path).values["config_dir"] == str(old)


def test_collapse_hides_without_changing_config_and_remembers(gui):
    before = gui.collect()
    section = gui.sections["Checkpoints"]
    section.header.click()
    assert section.content.isHidden()
    assert section.header.arrowType() == QtCore.Qt.ArrowType.RightArrow
    assert gui.collect() == before
    assert "Checkpoints" in Preferences(gui.preferences.path).values["collapsed_sections"]
    section.header.click()
    assert not section.content.isHidden()
    assert "Checkpoints" not in Preferences(gui.preferences.path).values["collapsed_sections"]


def test_preference_restore_and_launch_override(gui, tmp_path):
    custom = tmp_path / "custom"
    gui._set_config_dir(str(custom))
    gui.autosave_chk.setChecked(False)
    gui.sections["torch.compile"].setExpanded(False)
    other = gui_module.TrainingGUI(preferences=Preferences(gui.preferences.path))
    try:
        assert other.config_dir == custom
        assert not other.autosave_chk.isChecked()
        assert other.sections["torch.compile"].content.isHidden()
    finally:
        other.close()
    override = tmp_path / "override"
    other = gui_module.TrainingGUI(preferences=Preferences(gui.preferences.path), config_dir=str(override))
    try:
        assert other.config_dir == override
        assert Preferences(gui.preferences.path).values["config_dir"] == str(override)
    finally:
        other.close()
