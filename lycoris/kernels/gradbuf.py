"""One fp32 scratch allocation per kernel that emits several grads.

A kernel accumulating with atomics needs fp32 targets that start at zero, and
its caller needs them back in the parameter dtype. Packing every gradient of
one launch into a single allocation makes that one fill whatever the gradient
count is; the views alias that storage, so kernels take them as ordinary
tensors. The cast back is one launch per gradient into its own storage.
"""

import torch


class GradPack:
    """Zeroed fp32 views over one allocation, cast back in one launch."""

    def __init__(self, device, *shapes, zero: bool = True):
        self.shapes = shapes
        sizes = [int(torch.Size(s).numel()) for s in shapes]
        total = sum(sizes)
        make = torch.zeros if zero else torch.empty
        self.flat = make(total, device=device, dtype=torch.float32)
        self.views = []
        off = 0
        for size, shape in zip(sizes, shapes):
            self.views.append(self.flat[off : off + size].view(*shape))
            off += size

    def __iter__(self):
        return iter(self.views)

    def to(self, *dtypes):
        """Each gradient in its dtype, in its own storage (custom-op outputs may
        not alias each other); a single dtype applies to every view. fp32
        gradients are the views themselves."""
        if len(dtypes) == 1:
            dtypes = dtypes * len(self.views)
        return [
            v if d == torch.float32 else v.to(d) for v, d in zip(self.views, dtypes)
        ]

    def like(self):
        """An uninitialised twin, for the tuner's scratch runs."""
        twin = GradPack.__new__(GradPack)
        twin.shapes = self.shapes
        twin.flat = torch.empty_like(self.flat)
        twin.views = []
        off = 0
        for shape in self.shapes:
            size = int(torch.Size(shape).numel())
            twin.views.append(twin.flat[off : off + size].view(*shape))
            off += size
        return twin
