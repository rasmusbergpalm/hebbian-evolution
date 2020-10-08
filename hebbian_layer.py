from torch import nn
import torch as t
import shapeguard


class HebbianLayer(nn.Module):

    def __init__(self, n_in, n_out, activation_fn, learn_init=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.learn_init = learn_init

        if learn_init:
            self.W_init = nn.Parameter(t.randn((n_in, n_out)))
            self.W = self.W_init + 0
        else:
            self.W = t.randn((self.n_in, self.n_out))

        self.A = nn.Parameter(t.randn((n_in, n_out)))
        self.B = nn.Parameter(t.randn((n_in,)))
        self.C = nn.Parameter(t.randn((n_out,)))
        self.D = nn.Parameter(t.randn((n_in, n_out)))
        self.eta = nn.Parameter(t.randn((n_in, n_out)))
        self.activation_fn = activation_fn

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if self.learn_init:
            self.W = self.W_init + 0
        else:
            self.W = t.randn((self.n_in, self.n_out))

    def forward(self, pre):
        pre.sg((self.n_in,))
        post = self.activation_fn(pre @ self.W).sg((self.n_out,))
        self.update(pre, post)
        return post

    def update(self, pre, post):
        pre.sg((self.n_in,))
        post.sg((self.n_out,))

        self.W += self.eta * (
                self.A * (pre[:, None] @ post[None, :]).sg((self.n_in, self.n_out)) +
                (self.B * pre)[:, None].sg((self.n_in, 1)) +
                (self.C * post)[None, :].sg((1, self.n_out)) +
                self.D
        )
