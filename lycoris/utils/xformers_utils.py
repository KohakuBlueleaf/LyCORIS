memory_efficient_attention = None
try:
    import xformers
except Exception:
    pass

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_AVAIL = True
except Exception:
    memory_efficient_attention = None
    XFORMERS_AVAIL = False
