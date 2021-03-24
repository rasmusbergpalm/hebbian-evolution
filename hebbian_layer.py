import torch as t
import shapeguard


class HebbianLayer:

    def __init__(self, hebb_coeff: t.Tensor, activation_fn, weights: t.Tensor = None, normalize=False):
        self.n_in, self.n_out, _ = hebb_coeff.shape

        if weights is not None:
            weights.sg((self.n_in, self.n_out))
        else:
            weights = 0.2 * t.rand((self.n_in, self.n_out), requires_grad=False) - 0.1

        self.W = weights
        self.h = hebb_coeff
        self.normalize = normalize
        self.activation_fn = activation_fn

    def get_weights(self):
        return self.W + 0

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

        if self.normalize:
            self.W = self.W / self.W.abs().max()
