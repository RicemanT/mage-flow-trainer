"""Dataset source exclusivity and discoverable experimental controls."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from trainer.gui import bridge
from trainer.gui.app import TrainingGUI
from trainer.gui.widgets import STYLESHEET


@pytest.fixture
def gui(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(STYLESHEET)
    monkeypatch.setattr(TrainingGUI, "_load_presets", lambda self: None)
    window = TrainingGUI()
    window._apply(bridge.defaults() | {"dataset.path": "/images"})
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def test_subset_rows_disable_single_source_and_restore_it(gui):
    subsets = gui.editors["dataset.subsets"]
    subsets._add_row()
    assert not gui.editors["dataset.path"].widget.isEnabled()
    assert not gui.editors["dataset.num_repeats"].widget.isEnabled()
    assert gui.collect()["dataset.path"] is None
    assert not bridge.validate(gui.collect())[0]
    line = subsets.table.cellWidget(0, 0).findChild(QtWidgets.QLineEdit)
    line.setText("/other")
    assert gui.collect()["dataset.subsets"] == [{"path": "/other", "texture": False}]
    assert "dataset.path" not in bridge.prune_defaults(gui.collect())
    subsets._clear()
    assert gui.editors["dataset.path"].widget.isEnabled()
    assert gui.editors["dataset.num_repeats"].widget.isEnabled()
    assert gui.collect()["dataset.path"] == "/images"


def test_experimental_visibility_preserves_active_settings(gui):
    assert not gui.tab_bar.isTabVisible(gui.experimental_tab)
    gui.experimental_chk.setChecked(True)
    assert gui.tab_bar.isTabVisible(gui.experimental_tab)
    assert not gui.collect()["self_flow.enabled"]
    gui.experimental_chk.setChecked(False)
    for key in ["self_flow.enabled", "flow.dual_timestep", "rti.enabled",
                "perceptual.enabled", "adversarial.enabled"]:
        gui._apply(bridge.defaults() | {"dataset.path": "/images", key: True})
        assert gui.tab_bar.isTabVisible(gui.experimental_tab)
        assert not gui.experimental_chk.isEnabled()
        assert gui.collect()[key]
    gui._apply(bridge.defaults() | {"dataset.path": "/images"})
    assert gui.experimental_chk.isEnabled()
    gui.experimental_chk.setChecked(False)
    assert not gui.tab_bar.isTabVisible(gui.experimental_tab)


def test_subset_controls_have_room_at_small_window(gui):
    gui.resize(1100, 820)
    gui.editors["dataset.subsets"]._add_row()
    gui.show()
    QtWidgets.QApplication.processEvents()
    table = gui.editors["dataset.subsets"].table
    spin = table.cellWidget(0, 1)
    assert table.columnWidth(1) >= spin.sizeHint().width()
    assert table.rowHeight(0) >= spin.sizeHint().height() + 8
    spin.setValue(999)
    assert gui.collect()["dataset.subsets"] == []
    assert spin.value() == 999
