"""One fp32 scratch allocation per kernel that emits several grads.

A kernel accumulating with atomics needs fp32 targets that start at zero, and
its caller needs them back in the parameter dtype. Packing every gradient of
one launch into a single allocation makes that one fill and one cast whatever
the gradient count is; the views alias that storage, so kernels take them as
ordinary tensors.
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
        """Cast once, then re-split; a single dtype applies to every view."""
        if len(dtypes) == 1:
            dtypes = dtypes * len(self.views)
        if len(set(dtypes)) == 1:
            flat = self.flat.to(dtypes[0])
            out, off = [], 0
            for shape in self.shapes:
                size = int(torch.Size(shape).numel())
                out.append(flat[off : off + size].view(*shape))
                off += size
            return out
        return [v.to(d) for v, d in zip(self.views, dtypes)]

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
