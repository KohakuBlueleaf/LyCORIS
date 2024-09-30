from functools import cache

SUPPORT_QUANT = False
try:
    from bitsandbytes.nn import LinearNF4, Linear8bitLt, LinearFP4

    SUPPORT_QUANT = True
except Exception:
    import torch.nn as nn

    class LinearNF4(nn.Linear):
        pass

    class Linear8bitLt(nn.Linear):
        pass

    class LinearFP4(nn.Linear):
        pass


try:
    from quanto.nn import QLinear, QConv2d, QLayerNorm

    SUPPORT_QUANT = True
except Exception:
    import torch.nn as nn

    class QLinear(nn.Linear):
        pass

    class QConv2d(nn.Conv2d):
        pass

    class QLayerNorm(nn.LayerNorm):
        pass


try:
    from optimum.quanto.nn import (
        QLinear as QLinearOpt,
        QConv2d as QConv2dOpt,
        QLayerNorm as QLayerNormOpt,
    )

    SUPPORT_QUANT = True
except Exception:
    import torch.nn as nn

    class QLinearOpt(nn.Linear):
        pass

    class QConv2dOpt(nn.Conv2d):
        pass

    class QLayerNormOpt(nn.LayerNorm):
        pass


from ..logging import logger


QuantLinears = (
    Linear8bitLt,
    LinearFP4,
    LinearNF4,
    QLinear,
    QConv2d,
    QLayerNorm,
    QLinearOpt,
    QConv2dOpt,
    QLayerNormOpt,
)


@cache
def log_bypass():
    return logger.warning(
        "Using bnb/quanto/optimum-quanto with LyCORIS will enable force-bypass mode."
    )


@cache
def log_suspect():
    return logger.warning(
        "Non-native Linear detected but bypass_mode is not set. "
        "Automatically using force-bypass mode to avoid possible issues. "
        "Please set bypass_mode=False explicitly if there are no quantized layers."
    )
