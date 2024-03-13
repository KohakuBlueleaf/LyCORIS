try:
    from bitsandbytes.nn import (
        LinearNF4, Linear8bitLt, LinearFP4
    )
except:
    import torch.nn as nn
    class LinearNF4(nn.Linear):
        pass
    class Linear8bitLt(nn.Linear):
        pass
    class LinearFP4(nn.Linear):
        pass