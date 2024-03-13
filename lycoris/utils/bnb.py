from functools import cache

try:
    from bitsandbytes.nn import LinearNF4, Linear8bitLt, LinearFP4
except:
    import torch.nn as nn

    class LinearNF4(nn.Linear):
        pass

    class Linear8bitLt(nn.Linear):
        pass

    class LinearFP4(nn.Linear):
        pass


from ..logging import logger


QuantLinears = (Linear8bitLt, LinearFP4, LinearNF4)


@cache
def log_bypass():
    return logger.warning("Using bnb with LyCORIS will enable force-bypass mode.")
