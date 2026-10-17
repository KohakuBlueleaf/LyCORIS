"""Device time of candidate configs: interleaved rounds over L2-exceeding buffer sets, CUDA-graph replayed."""

from collections.abc import Callable

import torch

from .device import DeviceInfo


def _events(kind: str):
    mod = torch.cuda if kind == "cuda" else torch.xpu
    return mod.Event(enable_timing=True), mod.Event(enable_timing=True)


def _sync(kind: str) -> None:
    (torch.cuda if kind == "cuda" else torch.xpu).synchronize()


def buffer_sets(bytes_per_call: int, dev: DeviceInfo, cap: int = 64) -> int:
    """How many independent operand sets cycle L2 out: 4x L2, at least 1."""
    return max(1, min(cap, -(-4 * dev.l2_bytes // max(1, bytes_per_call))))


class _Batch:
    """One candidate's calls, captured as a graph in the shared pool when possible."""

    def __init__(self, calls: list[Callable[[], object]], dev: DeviceInfo, pool):
        self.calls = calls
        self.graph = None
        if pool is not None:
            try:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for call in calls:
                        call()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    for call in calls:
                        call()
                self.graph = graph
            except Exception:  # noqa: BLE001 - an uncapturable call is timed eagerly
                self.graph = None
                torch.cuda.synchronize()

    def run(self) -> None:
        if self.graph is not None:
            self.graph.replay()
        else:
            for call in self.calls:
                call()


def time_candidates(
    makers: list[Callable[[int], Callable[[], object]]],
    sets: int,
    dev: DeviceInfo,
    rounds: int = 4,
    use_graph: bool = True,
    prune: float = 2.0,
    errors: list[str | None] | None = None,
) -> list[float]:
    """Best-of-rounds ms per call per candidate, inf on failure.

    makers[i](j) is candidate i's call on buffer set j; candidates over prune x
    the leader stop after round one; errors, when given, receives each
    candidate's failure or None.
    """
    pool = torch.cuda.graph_pool_handle() if use_graph and dev.kind == "cuda" else None
    batches: list[_Batch | None] = []
    if errors is not None:
        errors[:] = [None] * len(makers)
    for i, make in enumerate(makers):
        try:
            calls = [make(j) for j in range(sets)]
            calls[0]()
            _sync(dev.kind)
            batches.append(_Batch(calls, dev, pool))
        except Exception as e:  # noqa: BLE001 - a failing candidate times as inf
            batches.append(None)
            if errors is not None:
                errors[i] = f"{type(e).__name__}: {str(e)[:300]}"
            _sync(dev.kind)
    best = [float("inf")] * len(makers)
    for round_ in range(rounds):
        leader = min(best)
        for i, batch in enumerate(batches):
            if batch is None or (round_ > 0 and best[i] > prune * leader):
                continue
            start, end = _events(dev.kind)
            batch.run()
            start.record()
            batch.run()
            end.record()
            _sync(dev.kind)
            best[i] = min(best[i], start.elapsed_time(end) / sets)
    # Free the graphs before their pool's memory is reused.
    del batches
    _sync(dev.kind)
    return best
