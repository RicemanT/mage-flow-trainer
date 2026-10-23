"""The real trainer's optional diagnostics must not blank Live Metrics."""
import os
import time
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
import torch
from PySide6 import QtWidgets

from trainer.gui.metrics import LiveMetricsWidget, STEP_RE
from trainer.training.config import TrainConfig


@pytest.mark.parametrize('auxiliary', [False, True])
def test_current_training_log_reaches_graphs(auxiliary, monkeypatch, capsys):
    from trainer.training.train import Trainer

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    train = TrainConfig(log_every=1)
    # Match Modan's HF + paired-texture logging, along with newer diagnostics.
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(gather=lambda x: x, device='cpu', is_main_process=True),
        cfg=SimpleNamespace(train=train, optimizer=SimpleNamespace(max_grad_norm=1.0)),
        scheduler=SimpleNamespace(get_last_lr=lambda: [0.0012, 0.0004]),
        progress_mode='plain', bar=None, global_step=9, total_steps=450,
        last_grad_norm=torch.tensor(0.05),
        last_stats=({'texture':0.001, 'ot_moved':0.8, 'lpips':0.02,
                     'd_loss':0.3, 'g_loss':0.2, 'self_flow_alignment':0.01} if auxiliary else {}),
        last_preserve=torch.tensor(0.001) if auxiliary else None,
        progress_pct=lambda: 2.0, _eta=lambda: '12m30s',
    )
    Trainer._log(trainer, torch.tensor(0.5), torch.tensor(0.1) if auxiliary else None,
                 2, time.time()-1)
    line = capsys.readouterr().out.strip()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = LiveMetricsWidget()
    try:
        widget.parse_and_update(line)
        widget._flush()
        assert widget.latest['step'] == 9
        assert widget.latest['loss'] == 0.5
        assert widget.latest['lrs'] == [0.0012, 0.0004]
        assert widget.latest['eta'] == '12m30s'
        if auxiliary:
            assert widget.latest['mse'] == 0.4
            assert widget.latest['hf'] == 0.1
            assert widget.latest['ot'] == 0.8
        for graph in widget.graphs.values():
            assert graph['widget'].lines[0]['data']
        assert '9/450' in widget.header.text()
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_legacy_and_unknown_auxiliary_fields():
    line = ('e3 step 1410/3000 (47%)  loss 0.0712 (mse 0.0650 + hf 0.0062) '
            'future_metric 0.3 ot 0.83 lr 2.71e-05 1.80s/it peak 8.1/8.2GB eta 47m34s')
    match = STEP_RE.match(line)
    assert match is not None
    assert match['hf'] == '0.0062'
    assert match['peak'] == '8.1/8.2'
    assert STEP_RE.match('saved output/run/epoch001 (280 tensors)') is None
    assert STEP_RE.match('e0 step 1/20 (5%) loss nan lr 1e-4 1.0s/it peak 8.0GB') is None


def test_elapsed_clock_survives_pause_clear_and_stops_on_finish(monkeypatch):
    from trainer.gui import metrics
    clock = [100.0]
    monkeypatch.setattr(metrics.time, 'monotonic', lambda: clock[0])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = LiveMetricsWidget()
    try:
        assert widget.elapsed_text() == '-'
        widget.start_run()
        clock[0] += 3661
        widget.pause_btn.setChecked(True)
        widget._set_header()
        assert '01:01:01' in widget.header.text()
        widget.clear_data()
        assert widget.elapsed_text() == '01:01:01'
        clock[0] += 2
        widget.finish_run()
        clock[0] += 1000
        assert widget.elapsed_text() == '01:01:03'
        assert not widget.elapsed_timer.isActive()
        widget.start_run()
        assert widget.elapsed_text() == '00:00:00'
        clock[0] += 100 * 3600
        assert widget.elapsed_text() == '100:00:00'
    finally:
        widget.finish_run()
        widget.deleteLater()
        app.processEvents()
