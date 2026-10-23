import pytest

from trainer.training.config import load_config


@pytest.mark.parametrize("matmul", ["true", "false", '"auto"'])
def test_frozen_adapter_matmul_requires_input_gradients(tmp_path, matmul):
    path = tmp_path / "config.toml"
    path.write_text(
        '[dataset]\npath="/unused"\n'
        '[adapter]\nkind="lycoris_lora"\n'
        f'[quant]\nmode="frozen"\nuse_quantized_matmul={matmul}\n'
    )
    if matmul == "true":
        with pytest.raises(ValueError, match="input-gradient path"):
            load_config(path)
    else:
        load_config(path)
