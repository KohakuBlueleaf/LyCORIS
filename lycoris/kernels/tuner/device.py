"""Device facts the adapter kernels and the cost model read, queried from the driver."""

import ctypes
import ctypes.util
import functools
import re
import sys
from dataclasses import dataclass

import torch

_DEFAULT_L2 = 32 * 1024 * 1024
_DEFAULT_SMEM = 64 * 1024

# CUdevice_attribute values from cuda.h.
_CU_ATTR = {
    "units": 16,
    "l2_bytes": 38,
    "max_threads_sm": 39,
    "smem_sm": 81,
    "regs_sm": 82,
    "smem": 97,
    "max_blocks_sm": 106,
    "smem_reserved": 111,
}


@dataclass(frozen=True)
class DeviceInfo:
    """One GPU as the kernels see it; kind is "cuda" or "xpu", arch keys the tuning table."""

    kind: str
    arch: str
    name: str
    index: int
    units: int
    capability: tuple[int, int] | None
    l2_bytes: int
    smem: int
    bf16: bool
    mma: bool
    regs_sm: int = 65536
    smem_sm: int = _DEFAULT_SMEM
    smem_reserved: int = 0
    max_threads_sm: int = 2048
    max_blocks_sm: int = 32

    @property
    def async_copy(self) -> bool:
        """cp.async staging (sm_80+): pipeline stages keep loads in flight."""
        return self.capability is not None and self.capability >= (8, 0)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@functools.cache
def _libcuda():
    name = (
        "nvcuda.dll"
        if sys.platform == "win32"
        else (ctypes.util.find_library("cuda") or "libcuda.so.1")
    )
    try:
        lib = ctypes.CDLL(name)
    except OSError:
        return None
    return lib if lib.cuInit(0) == 0 else None


def _driver_attrs(index: int) -> dict[str, int]:
    """Occupancy limits from cuDeviceGetAttribute; empty without the driver library."""
    lib = _libcuda()
    if lib is None:
        return {}
    dev = ctypes.c_int()
    if lib.cuDeviceGet(ctypes.byref(dev), index) != 0:
        return {}
    out = {}
    for field, attr in _CU_ATTR.items():
        value = ctypes.c_int()
        if lib.cuDeviceGetAttribute(ctypes.byref(value), attr, dev) == 0:
            out[field] = value.value
    return out


@functools.cache
def _cuda_info(index: int) -> DeviceInfo:
    props = torch.cuda.get_device_properties(index)
    l2 = getattr(props, "L2_cache_size", 0) or _DEFAULT_L2
    smem = getattr(props, "shared_memory_per_block_optin", 0) or _DEFAULT_SMEM
    cap = (props.major, props.minor)
    attrs = _driver_attrs(index)
    return DeviceInfo(
        kind="cuda",
        arch=f"sm_{props.major}{props.minor}",
        name=props.name,
        index=index,
        units=attrs.get("units", props.multi_processor_count),
        capability=cap,
        l2_bytes=attrs.get("l2_bytes", l2),
        smem=attrs.get("smem", smem),
        bf16=cap >= (8, 0),
        # Triton emits mma.sync from sm_80 only.
        mma=cap >= (8, 0),
        regs_sm=attrs.get("regs_sm", getattr(props, "regs_per_multiprocessor", 65536)),
        smem_sm=attrs.get(
            "smem_sm", getattr(props, "shared_memory_per_multiprocessor", smem)
        ),
        smem_reserved=attrs.get("smem_reserved", 0),
        max_threads_sm=attrs.get(
            "max_threads_sm", getattr(props, "max_threads_per_multi_processor", 2048)
        ),
        max_blocks_sm=attrs.get("max_blocks_sm", 32),
    )


@functools.cache
def _xpu_info(index: int) -> DeviceInfo:
    props = torch.xpu.get_device_properties(index)
    units = getattr(props, "gpu_subslice_count", 0) or getattr(
        props, "max_compute_units", 1
    )
    slm = getattr(props, "local_mem_size", 0) or _DEFAULT_SMEM
    return DeviceInfo(
        kind="xpu",
        arch=f"xpu-{_slug(props.name)}",
        name=props.name,
        index=index,
        units=units,
        capability=None,
        l2_bytes=getattr(props, "last_level_cache_size", 0) or _DEFAULT_L2,
        smem=slm,
        bf16=True,
        mma=bool(getattr(props, "has_subgroup_matrix_multiply_accumulate", False)),
        smem_sm=slm,
        max_threads_sm=getattr(props, "max_work_group_size", 0) or 1024,
        max_blocks_sm=64,
    )


def device_info(device: torch.device) -> DeviceInfo:
    index = device.index
    if device.type == "cuda":
        return _cuda_info(torch.cuda.current_device() if index is None else index)
    if device.type == "xpu":
        return _xpu_info(torch.xpu.current_device() if index is None else index)
    raise ValueError(f"no fused kernels for device type {device.type}")
