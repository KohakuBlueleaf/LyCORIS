import torch
import torch.nn as nn

from .locon import LoConModule


class LiLoConModule(LoConModule):
    def make_lightweight(self, down, up, seed=None, down_aux=None, up_aux=None):
        if down.dim() == 3:
            down = down.reshape(down.size(0), self.lora_dim, -1)
            up = up.reshape(up.size(0), -1, self.lora_dim)
        else:
            down = down.reshape(self.lora_dim, -1)
            up = up.reshape(-1, self.lora_dim)
        # print(up.shape)
        if seed is None:
            assert down_aux is not None and up_aux is not None
        else:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
            if down_aux is None or up_aux is None:
                down_aux = torch.empty(
                    down.size(down.dim() - 1),
                    self.lora_down.weight.size(1),
                    device=down.device,
                )
                up_aux = torch.empty(
                    self.lora_up.weight.size(0), up.size(up.dim() - 2), device=up.device
                )
                nn.init.orthogonal_(down_aux)
                nn.init.orthogonal_(up_aux)
                # print(up_aux.shape)
            torch.set_rng_state(rng_state)
        if down.dim() == 3 and down.size(0) == 1:
            down = down.squeeze(0)
        if up.dim() == 3 and up.size(0) == 1:
            up = up.squeeze(0)
        down = down + 1  # avoid zero grad or slow training, give it a constant
        return (down @ down_aux), (up_aux @ up)

    def apply_lightweight(self, down, up, seed=None, down_aux=None, up_aux=None):
        down_weight, up_weight = self.make_lightweight(down, up, seed, down_aux, up_aux)
        self.lora_down.weight.data = down_weight
        self.lora_up.weight.data = up_weight
        return down_weight, up_weight

    def hypernet_forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.scale * self.multiplier

        down_weight = self.down
        up_weight = self.up

        x_batch = None
        if down_weight.dim() == 3:
            if x.size(0) != down_weight.size(0):
                assert (
                    self.isconv == False
                ), "Convolutional hypernet with batch size mismatch is not supported"
                x_batch = x.size(0)
                x = x.view(down_weight.size(0), -1, *x.shape[1:])

            if self.isconv:
                mid = torch.einsum("ijk, ik... -> ij...", down_weight, x)
            else:
                mid = torch.einsum("ijk, i...k -> i...j", down_weight, x)
        else:
            if self.isconv:
                weight = down_weight.unsqueeze(-1).unsqueeze(-1)
            else:
                weight = down_weight
            mid = self.down_op(x, weight)

        if self.rank_dropout and self.training:
            drop = (
                torch.rand(self.lora_dim, device=mid.device) < self.rank_dropout
            ).to(mid.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            if (dims := len(x.shape)) == 4:
                drop = drop.view(1, -1, 1, 1)
            else:
                drop = drop.view(*[1] * (dims - 1), -1)
            mid = mid * drop

        if up_weight.dim() == 3:
            mid_batch = None
            if mid.size(0) != up_weight.size(0):
                assert (
                    self.isconv == False
                ), "Convolutional hypernet with batch size mismatch is not supported"
                mid_batch = mid.size(0)
                mid = mid.view(up_weight.size(0), -1, *mid.shape[1:])

            if self.isconv:
                up = torch.einsum("ijk, ik... -> ij...", up_weight, mid)
            else:
                up = torch.einsum("ijk, i...k -> i...j", up_weight, mid)

            if mid_batch is not None:
                up = up.view(mid_batch, *up.shape[2:])
        else:
            if self.isconv:
                weight = up_weight.unsqueeze(-1).unsqueeze(-1)
            else:
                weight = up_weight
            up = self.up_op(mid, weight)

        if x_batch is not None:
            up = up.view(x_batch, *up.shape[2:])

        org_out = self.org_forward(x)
        # print(x.shape, org_out.shape, up.shape)
        return org_out + self.dropout(up) * scale
