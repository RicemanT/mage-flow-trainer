"""Live, deterministic probability preview; no sampling or training RNG consumption."""
import math
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from ..training.flow import FlowConfig, get_lin_function, sample_timesteps
from ..training.moss_timesteps import probability_table
from .widgets import THEME


SAMPLER_KEYS = ("timestep_sample_method", "sigmoid_scale", "shift", "flux_shift",
                "moss_mu", "moss_std", "moss_cutoff", "moss_uniform_mix", "moss_p_full", "moss_steps")


@dataclass(frozen=True)
class Distribution:
    probabilities: tuple[float, ...]
    median: float
    high_noise: float


def distribution(cfg, latent_h=64, latent_w=64, bins=40):
    """Exact bin integrals (or sums for the discrete recipe), before curriculum mapping."""
    if cfg.timestep_sample_method == "moss":
        p = probability_table(cfg.moss_mu, cfg.moss_std, cfg.moss_cutoff,
                              cfg.moss_uniform_mix, cfg.moss_p_full, cfg.moss_steps).tolist()
        masses = [0.0] * bins
        high = 0.0
        for j, mass in enumerate(p, 1):
            # Integer arithmetic avoids moving a grid point across a bin edge by rounding.
            masses[min(j * bins // cfg.moss_steps, bins - 1)] += mass
            if 10 * j >= 9 * cfg.moss_steps:
                high += mass
        median = sample_timesteps(cfg, 1, latent_h, latent_w, "cpu", quantile=.5).item()
    else:
        shift = cfg.shift if cfg.shift is not None else (
            math.exp(get_lin_function(y1=.5, y2=1.15)(latent_h * latent_w)) if cfg.flux_shift else 1)
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("Shift must be finite and positive")

        def cdf(t):
            if t <= 0 or t >= 1:
                return float(t >= 1)
            # Inverse of the trainer's static shift (Flux shift has sigma=1).
            base = t / (shift + (1 - shift) * t)
            if cfg.timestep_sample_method == "uniform":
                return base
            z = (math.log(base) - math.log1p(-base)) / cfg.sigmoid_scale
            return .5 * math.erfc(-z / math.sqrt(2))

        edges = [cdf(j / bins) for j in range(bins + 1)]
        masses = [max(0, b - a) for a, b in zip(edges, edges[1:])]
        median = shift / (1 + shift)
        high = 1 - cdf(.9)
    return Distribution(tuple(masses), median, high)


class _Canvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.error = ""
        self.setMinimumSize(260, 220)
        self.setMouseTracking(True)

    def graph_rect(self):
        return QtCore.QRectF(45, 27, max(1, self.width() - 60), max(1, self.height() - 72))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QColor(THEME.text_muted))
        if self.result is None:
            painter.drawText(self.rect().adjusted(8, 8, -8, -8),
                             QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
                             self.error or "Select timestep settings to preview")
            return
        r = self.graph_rect()
        values = self.result.probabilities
        ceiling = max(values) * 100 * 1.1
        painter.drawText(QtCore.QRectF(0, 0, self.width(), 23),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Probability per bin (%)")
        for i in range(5):
            y = r.bottom() - i * r.height() / 4
            painter.setPen(QtGui.QColor(THEME.text_muted))
            painter.drawText(QtCore.QRectF(0, y - 9, 39, 18), QtCore.Qt.AlignmentFlag.AlignRight,
                             f"{ceiling * i / 4:.1f}")
            pen = QtGui.QColor(THEME.text_muted)
            pen.setAlpha(50)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(r.left(), y), QtCore.QPointF(r.right(), y))
        width = r.width() / len(values)
        for i, probability in enumerate(values):
            height = probability * 100 / ceiling * r.height()
            painter.fillRect(QtCore.QRectF(r.left() + i * width, r.bottom() - height,
                                           max(.5, width - 1), height), QtGui.QColor(THEME.accent))
        painter.setPen(QtGui.QColor(THEME.text_muted))
        for i in range(6):
            x = r.left() + i * r.width() / 5
            painter.drawText(QtCore.QRectF(x - 20, r.bottom() + 3, 40, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, f"{i / 5:.1f}")
        painter.drawText(QtCore.QRectF(0, self.height() - 22, self.width(), 22),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Clean (t=0)  →  Pure noise (t=1)")

    def mouseMoveEvent(self, event):
        r = self.graph_rect()
        if self.result and r.contains(event.position()):
            n = len(self.result.probabilities)
            i = min(n - 1, int((event.position().x() - r.left()) / r.width() * n))
            right = "]" if i == n - 1 else ")"
            QtWidgets.QToolTip.showText(event.globalPosition().toPoint(),
                f"t ∈ [{i/n:.3f}, {(i+1)/n:.3f}{right}: "
                f"{100*self.result.probabilities[i]:.3f}% of draws", self)
        else:
            QtWidgets.QToolTip.hideText()


class TimestepDistributionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._flat = {}
        self._last = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.grid_controls = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.grid_controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel("Preview latent H × W"))
        self.height_spin, self.width_spin = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin in (self.height_spin, self.width_spin):
            spin.setRange(1, 512)
            spin.setValue(64)
            spin.valueChanged.connect(lambda _: self.set_config(self._flat))
            row.addWidget(spin)
        self.grid_controls.setToolTip("Preview only; does not change training resolution. "
                                     "Resolution-aware shift uses the actual latent grid for each training image.")
        layout.addWidget(self.grid_controls)
        self.canvas = _Canvas()
        layout.addWidget(self.canvas)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        note = QtWidgets.QLabel("Sampling frequency, not loss weight. Before curriculum range mapping. "
                                "Hover a bar for its probability; vertical axis auto-scales.")
        note.setWordWrap(True)
        layout.addWidget(note)

    def set_config(self, flat):
        self._flat = flat
        options = {key: flat[f"flow.{key}"] for key in SAMPLER_KEYS if f"flow.{key}" in flat}
        h, w = self.height_spin.value(), self.width_spin.value()
        signature = (tuple(options.items()), h, w)
        if signature == self._last:
            return
        self._last = signature
        self.grid_controls.setVisible(bool(options.get("flux_shift")))
        try:
            result = distribution(FlowConfig(**options), h, w)
            self.canvas.result = result
            self.canvas.error = ""
            self.summary.setText(f"Median t: {result.median:.3f}   •   "
                                 f"High noise (t ≥ 0.9): {result.high_noise:.1%}")
        except (ValueError, TypeError, OverflowError) as exc:
            self.canvas.result = None
            self.canvas.error = str(exc)
            self.summary.setText("Fix the timestep settings to update the preview.")
        self.canvas.update()
