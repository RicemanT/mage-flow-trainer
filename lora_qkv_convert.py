#!/usr/bin/env python
"""Convert MOSS LoRA/LoCon joint Q/K/V layout without retraining or changing strength.

No arguments opens a small GUI, like lora_alpha_edit.py. CLI examples:
    python lora_qkv_convert.py model.safetensors --format fused
    python lora_qkv_convert.py model-qkv-fused.safetensors --format separated

Creates a new file; never overwrites the source or an existing destination.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile


def convert_file(source, destination=None, layout='fused'):
    from safetensors import SafetensorError, safe_open
    from safetensors.torch import load_file, save_file
    from trainer.modeling.moss_export import fuse_qkv, separate_adapter_qkv

    if layout not in ('fused', 'separated'):
        raise ValueError('Choose fused or separated')
    source = Path(source).resolve()
    destination = Path(destination).resolve() if destination else source.with_name(source.stem+f'-qkv-{layout}.safetensors')
    if destination == source or destination.exists():
        raise ValueError('Choose a new output filename; existing files are never overwritten')
    try:
        state = load_file(str(source),device='cpu')
        with safe_open(source,framework='pt') as reader:
            metadata = dict(reader.metadata() or {})
    except SafetensorError as exc:
        raise ValueError(f'Cannot read LoRA checkpoint: {exc}') from exc
    state, details = (fuse_qkv(state,adapter=True) if layout == 'fused' else separate_adapter_qkv(state))
    if details['joint_blocks'] == 0:
        raise ValueError(f'No convertible MOSS joint Q/K/V adapters found for {layout} output; the file may already use that format')
    metadata['moss_qkv_format'] = layout
    metadata['moss_qkv_conversion'] = json.dumps(details)
    metadata['inference_conversion'] = json.dumps(dict(source=source.name,layout=layout,**details))
    fd, temporary = tempfile.mkstemp(prefix=destination.name+'.',suffix='.tmp',dir=destination.parent)
    os.close(fd)
    try:
        save_file(state,temporary,metadata=metadata)
        # Atomic publication without replacing an existing destination.
        os.link(temporary,destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination, details


def gui(initial=None):
    from PySide6 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    window.setWindowTitle('MOSS LoRA Q/K/V converter')
    layout = QtWidgets.QVBoxLayout(window)
    info = QtWidgets.QLabel('Convert existing MOSS LoRA/LoCon files. Fused output supports fused-base loaders such as Forge.\nScaling is preserved; files can grow. LoKr and DoRA Q/K/V conversion is unsupported.')
    info.setWordWrap(True)
    layout.addWidget(info)
    row = QtWidgets.QHBoxLayout()
    path = QtWidgets.QLineEdit(str(initial or ''))
    browse = QtWidgets.QPushButton('Choose LoRA…')
    row.addWidget(path); row.addWidget(browse); layout.addLayout(row)
    choice = QtWidgets.QComboBox(); choice.addItems(['fused','separated'])
    layout.addWidget(choice)
    save = QtWidgets.QPushButton('Save converted copy…'); layout.addWidget(save)
    status = QtWidgets.QLabel('Original file is preserved.'); status.setWordWrap(True); layout.addWidget(status)
    def pick():
        selected,_ = QtWidgets.QFileDialog.getOpenFileName(window,'Choose LoRA',path.text(),'Safetensors (*.safetensors)')
        if selected: path.setText(selected)
    def write():
        source = Path(path.text())
        if not source.is_file():
            status.setText('Choose an existing LoRA file.'); return
        target = source.with_name(source.stem+f'-qkv-{choice.currentText()}.safetensors')
        selected,_ = QtWidgets.QFileDialog.getSaveFileName(window,'Save a new converted copy',str(target),'Safetensors (*.safetensors)')
        if not selected: return
        save.setEnabled(False)
        app.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        status.setText('Converting…'); app.processEvents()
        try:
            out, details = convert_file(source,selected,choice.currentText())
            status.setText(f"Saved {details['joint_blocks']} joint blocks: {out}")
        except (ValueError,OSError,RuntimeError) as exc:
            status.setText(str(exc))
        finally:
            app.restoreOverrideCursor(); save.setEnabled(True)
    browse.clicked.connect(pick); save.clicked.connect(write)
    window.resize(650,230); window.show()
    return app.exec()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input',nargs='?',type=Path)
    parser.add_argument('--output','-o',type=Path)
    parser.add_argument('--format',choices=('fused','separated'),default='fused')
    parser.add_argument('--gui',action='store_true')
    args = parser.parse_args(argv)
    if args.gui or args.input is None:
        return gui(args.input)
    try:
        out,details = convert_file(args.input,args.output,args.format)
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(1,f'Conversion failed: {exc}\n')
    print(f"Saved {out} ({details['joint_blocks']} joint blocks; original preserved)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
