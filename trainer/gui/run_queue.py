"""Ordered TOML training queue. Config files remain owned by the user."""
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .widgets import make_btn, usable_dialog_start


class RunQueueWidget(QtWidgets.QWidget):
    startRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            "Add saved TOMLs in the order to run. Each config is validated, cached, then trained "
            "with the selected GPUs. The queue stops on any error. Stop cancels the remaining runs. "
            "Unsaved editor changes are not included; relative paths use the trainer folder.")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        layout.addWidget(self.list, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.edit_buttons = [
            make_btn("Add TOMLs", self._add),
            make_btn("Move up", lambda: self.move(-1)),
            make_btn("Move down", lambda: self.move(1)),
            make_btn("Remove selected", self.remove_selected),
            make_btn("Clear queue", self.list.clear),
        ]
        for button in self.edit_buttons:
            buttons.addWidget(button)
        buttons.addStretch()
        self.start_btn = make_btn("Start queue", self.startRequested.emit, style="accent")
        buttons.addWidget(self.start_btn)
        layout.addLayout(buttons)
        self.status = QtWidgets.QLabel("No runs queued.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        for signal in (self.list.model().rowsInserted, self.list.model().rowsRemoved,
                       self.list.model().modelReset, self.list.model().rowsMoved):
            signal.connect(self._changed)
        self._changed()

    def _changed(self, *args):
        self.start_btn.setEnabled(not self._busy and self.list.count() > 0)
        if not self._busy:
            self.status.setText(f"{self.list.count()} configuration(s) queued.")

    def paths(self):
        return [Path(self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole))
                for i in range(self.list.count())]

    def add_paths(self, paths):
        existing = set(self.paths())
        for path in map(lambda p: Path(p).resolve(), paths):
            if path in existing:
                continue
            item = QtWidgets.QListWidgetItem(str(path))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.list.addItem(item)
            existing.add(path)

    def _add(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add training configs", usable_dialog_start("configs"), "TOML configs (*.toml)")
        self.add_paths(paths)

    def remove_selected(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def move(self, direction):
        rows = sorted((self.list.row(item) for item in self.list.selectedItems()), reverse=direction > 0)
        selected = set(rows)
        for row in rows:
            target = row + direction
            if 0 <= target < self.list.count() and target not in selected:
                item = self.list.takeItem(row)
                self.list.insertItem(target, item)
                item.setSelected(True)
                selected.remove(row)
                selected.add(target)

    def set_busy(self, busy):
        self._busy = busy
        self.list.setEnabled(not busy)
        for button in self.edit_buttons:
            button.setEnabled(not busy)
        self.start_btn.setEnabled(not busy and self.list.count() > 0)

    def reset_status(self):
        for i, path in enumerate(self.paths()):
            self.list.item(i).setText(f"Pending — {path}")

    def mark(self, index, status):
        path = self.paths()[index]
        self.list.item(index).setText(f"{status} — {path}")
        self.status.setText(f"{index + 1}/{self.list.count()}: {status} — {path.name}")

    def end(self, message):
        for i in range(self.list.count()):
            if self.list.item(i).text().startswith("Pending —"):
                self.mark(i, "Skipped")
        self.set_busy(False)
        self.status.setText(message)
