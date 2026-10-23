"""Global sample weighting and coordinated nonfinite-update decisions."""
import torch
import torch.distributed as dist


def global_sample_scale(local_samples, accumulation_steps, device):
    """Undo Accelerate's accumulation divisor and DDP's rank mean.

    Each backward receives the SUM of per-image losses in that microbatch.
    After reduction, divide by the actual image count over the whole update.
    This also handles partial accumulation groups and uneven bucket batches.
    """
    count = torch.tensor(local_samples, device=device, dtype=torch.int64)
    world = 1
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(count)
    if count.item() <= 0:
        raise ValueError('An optimizer update must contain at least one image')
    return world * accumulation_steps / count


def scale_gradients(parameters, scale):
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scale)


def gradient_norm(parameters, device):
    norms = [torch.linalg.vector_norm(p.grad.detach().float())
             for p in parameters if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.zeros((), device=device)


def globally_finite(loss_finite, norm):
    bad = (~loss_finite.bool() | ~torch.isfinite(norm)).to(torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    return not bool(bad.item())


def nonfinite_names(named_parameters, limit=8):
    names = []
    for name, p in named_parameters:
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            names.append(name)
            if len(names) == limit:
                break
    return names


def guard_released_gradient(parameter):
    """Installed BEFORE optimizer hooks: stop, never continue a partial update."""
    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
        raise FloatingPointError(
            'Nonfinite gradient during gradient release. Training stopped; earlier parameters '
            'may already have updated, so this partial step cannot be skipped or saved safely.')
