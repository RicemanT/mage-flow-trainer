import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from trainer.training.step_safety import global_sample_scale, scale_gradients


def test_partial_accumulation_matches_mean_of_actual_images():
    acc = Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    batches = [torch.tensor([[1.], [2.], [3.]]), torch.tensor([[10.]]), torch.tensor([[5.]])]
    model, optimizer, loader = acc.prepare(model, optimizer, DataLoader(batches, batch_size=None))
    gradients, count = [], 0
    for batch in loader:
        with acc.accumulate(model):
            count += len(batch)
            acc.backward(model(batch).mean() * len(batch))
            if acc.sync_gradients:
                scale_gradients(model.parameters(), global_sample_scale(count, 2, acc.device))
                gradients.append(model.weight.grad.item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                count = 0
    assert gradients == [4., 5.]
    assert model.weight.item() == -9.
    acc.free_memory()
