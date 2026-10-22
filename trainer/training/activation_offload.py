"""Bounded offload of checkpoint boundary inputs, not the full autograd graph.

Copies are synchronous on the way to host so inputs can immediately be released.
Optional pinned-host prefetch overlaps the next boundary's restore with backward.
The budget counts live host tensor payloads, including shared saved views once;
allocator caching and CUDA staging buffers are not part of that payload budget.
"""

from contextlib import contextmanager
import math
import threading
import weakref

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint


def _storage_key(tensor):
    try:
        return tensor.device, tensor.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError):
        # Wrapper tensors (including quantized parameter wrappers) may not expose
        # ordinary storage. They cannot be offloaded by this implementation.
        return None


class _SavedInput:
    def __init__(self, manager, host, source, block):
        self.host, self.device, self.block = host, source.device, block
        self.source = weakref.ref(source)
        self.nbytes = host.numel() * host.element_size()
        self._manager = weakref.ref(manager)

    def __del__(self):
        manager = self._manager()
        if manager is not None:
            with manager._lock:
                manager.live_bytes -= self.nbytes


class ActivationOffloader:
    """One manager per model/rank, with no strong references to source tensors."""

    def __init__(self, model, budget_bytes, *, pin_memory=False, prefetch=False):
        if isinstance(budget_bytes, bool) or int(budget_bytes) != budget_bytes or budget_bytes <= 0:
            raise ValueError("activation offload budget must be a positive byte count")
        self._model = weakref.ref(model)
        self.budget_bytes = int(budget_bytes)
        self.pin_memory = bool(pin_memory)
        self.prefetch = bool(prefetch)
        self.live_bytes = self.peak_bytes = self.offloaded_tensors = self.budget_fallbacks = 0
        self._lock = threading.Lock()
        self._saved = weakref.WeakValueDictionary()
        self._blocks = {}
        self._next_block = 0
        self._scope_block = None
        self._excluded = set()
        self._prefetched = {}
        self._prefetch_block = None
        self._streams = {}

    def __deepcopy__(self, memo):
        if self.live_bytes or self._scope_block is not None:
            raise RuntimeError("cannot copy a model with live activation offload checkpoints")
        model = self._model()
        copied_model = memo.get(id(model))
        if copied_model is None:
            raise RuntimeError("copy the owning model, not its activation offload manager")
        copied = type(self)(copied_model, self.budget_bytes,
                            pin_memory=self.pin_memory, prefetch=self.prefetch)
        memo[id(self)] = copied
        return copied

    def _refresh_exclusions(self):
        model = self._model()
        self._excluded = set() if model is None else {
            key for tensor in (*model.parameters(), *model.buffers())
            if (key := _storage_key(tensor)) is not None
        }

    def _eligible(self, tensor):
        return tensor.device.type == "cuda" and tensor.layout == torch.strided and tensor.numel() > 0

    def pack(self, tensor):
        if not self._eligible(tensor):
            return tensor
        storage = _storage_key(tensor)
        if storage is None or storage in self._excluded:
            return tensor
        # Version distinguishes an in-place-modified value in shared storage.
        key = (storage, tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
               tensor.dtype, tensor._version)
        saved = self._saved.get(key)
        # Allocator addresses can be reused once the original tensor dies; its
        # weak identity prevents treating a later allocation as the old value.
        if saved is not None and saved.source() is not None:
            self._blocks[self._scope_block].append(weakref.ref(saved))
            return saved
        nbytes = tensor.numel() * tensor.element_size()
        with self._lock:
            if self.live_bytes + nbytes > self.budget_bytes:
                self.budget_fallbacks += 1
                return tensor
            self.live_bytes += nbytes
        try:
            host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=self.pin_memory)
            host.copy_(tensor.detach(), non_blocking=False)
            saved = _SavedInput(self, host, tensor, self._scope_block)
        except BaseException:
            with self._lock:
                self.live_bytes -= nbytes
            raise
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        self.offloaded_tensors += 1
        self._saved[key] = saved
        self._blocks[self._scope_block].append(weakref.ref(saved))
        return saved

    def _clear_prefetch(self):
        # A dropped prefetched tensor may still be being filled by the transfer
        # stream. record_stream preserves its storage until that copy completes.
        self._prefetched.clear()
        self._prefetch_block = None

    def _prefetch_previous(self, block):
        previous = block - 1
        if self._prefetch_block == previous:
            return
        self._clear_prefetch()
        refs = self._blocks.get(previous, ())
        for ref in refs:
            saved = ref()
            if saved is None or saved.device.type != "cuda" or id(saved) in self._prefetched:
                continue
            stream = self._streams.get(saved.device)
            if stream is None:
                stream = self._streams[saved.device] = torch.cuda.Stream(device=saved.device)
            with torch.cuda.stream(stream):
                restored = saved.host.to(saved.device, non_blocking=True)
                restored.record_stream(stream)
                ready = torch.cuda.Event()
                ready.record(stream)
            self._prefetched[id(saved)] = (restored, ready)
        self._prefetch_block = previous

    def unpack(self, saved, block=None):
        if isinstance(saved, torch.Tensor):
            return saved
        cached = self._prefetched.pop(id(saved), None)
        if cached is None:
            restored = saved.host.to(saved.device, non_blocking=False)
        else:
            restored, ready = cached
            stream = torch.cuda.current_stream(saved.device)
            stream.wait_event(ready)
            restored.record_stream(stream)
        # Do not replace the current block's pending prefetch on its first input:
        # only start another block once all prefetched inputs have been consumed.
        if self.prefetch and self.pin_memory and not self._prefetched:
            self._prefetch_previous(saved.block if block is None else block)
        return restored

    @contextmanager
    def scope(self):
        if self._scope_block is not None:
            raise RuntimeError("nested activation offload checkpoints are unsupported")
        if not self._saved:
            self._blocks.clear()
            self._clear_prefetch()
            self._next_block = 0
            self._refresh_exclusions()
        block = self._next_block
        self._next_block += 1
        self._scope_block = block
        self._blocks[block] = []

        def restore(saved):
            # A shared tensor can belong to many boundaries: choose the previous
            # boundary of this unpack, not the payload's first owner.
            return self.unpack(saved, block)

        try:
            with torch.autograd.graph.saved_tensors_hooks(self.pack, restore):
                yield
        finally:
            self._scope_block = None

    def checkpoint(self, function, *args):
        # Checkpoint otherwise keeps tensors nested inside tuples/dicts alive in
        # Python closures. Flatten first so every boundary tensor uses the hooks.
        leaves, spec = tree_flatten(args)

        def call(*flat_args):
            return function(*tree_unflatten(flat_args, spec))

        with self.scope():
            return checkpoint(call, *leaves, use_reentrant=False)


def configure_activation_offload(model, *, enabled=False, budget_mb=1024,
                                 pin_memory=False, prefetch=False,
                                 gradient_checkpointing=True, compile_mode=None):
    if enabled and not gradient_checkpointing:
        raise ValueError("activation offload requires gradient checkpointing")
    if enabled and compile_mode is not None:
        raise ValueError("activation offload currently requires compile disabled")
    if enabled and (not math.isfinite(budget_mb) or budget_mb <= 0):
        raise ValueError("activation offload budget must be positive and finite")
    model._activation_offloader = (
        ActivationOffloader(model, int(budget_mb * 1024 * 1024), pin_memory=pin_memory, prefetch=prefetch)
        if enabled else None
    )


def checkpoint_with_offload(model, function, *args):
    offloader = getattr(model, "_activation_offloader", None)
    if offloader is None:
        return checkpoint(function, *args, use_reentrant=False)
    return offloader.checkpoint(function, *args)
