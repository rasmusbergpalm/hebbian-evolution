from torch import nn
import torch as t
import shapeguard


class HebbianLayer(nn.Module):

    def __init__(self, n_in, n_out, activation_fn, learn_init=False):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.learn_init = learn_init

        if learn_init:
            self.W_init = nn.Parameter(t.randn((n_in, n_out)), requires_grad=False)
            self.W = self.W_init + 0
        else:
            self.W = t.randn((self.n_in, self.n_out))

        self.h = nn.Parameter(t.randn((n_in, n_out, 5)))
        self.activation_fn = activation_fn

    def get_weights(self):
        return self.W + 0

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

        eta, A, B, C, D = [v.squeeze().sg((self.n_in, self.n_out)) for v in self.h.split(1, -1)]
        self.W += eta * (
                A * (pre[:, None] @ post[None, :]).sg((self.n_in, self.n_out)) +
                (B * pre[:, None]).sg((self.n_in, self.n_out)) +
                (C * post[None, :]).sg((self.n_in, self.n_out)) +
                D
        )
