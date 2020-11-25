import matplotlib.pyplot as plt
import torch as t
import numpy as np

(c1, c2, h1, h2, h3, lrm) = t.load("results_4205277_best.t", map_location=t.device('cpu'))
lrm = lrm.detach().numpy()


def plot_dist(h):
    plt.figure(dpi=200)
    plt.hist(h.argmax(dim=-1).flatten().numpy(), bins=np.arange(17) - 0.5, density=True, rwidth=0.8)
    plt.title("h1")
    plt.show()
    plt.close()


def plot_lrm_imshow():
    plt.figure(dpi=200)
    plt.imshow(lrm)
    plt.colorbar()
    plt.show()
    plt.close()


def plot_lrm_5x5():
    plt.figure(dpi=200)
    fig, ax = plt.subplots(5, 5)
    for i in range(5):
        for j in range(i, 5):
            for m in range(lrm.shape[0]):
                ax[i, j].plot(lrm[m, i], lrm[m, j], '.')
    plt.show()
    plt.close()


def plot_lrm_5x1():
    plt.figure(dpi=200)
    titles = ['eta', 'A', 'B', 'C', 'D']
    fig, ax = plt.subplots(5, 1, squeeze=True)
    for i in range(5):
        ax[i].set_title(titles[i])
        for m in range(lrm.shape[0]):
            ax[i].plot(lrm[m, i], 0, '.')
    plt.show()
    plt.close()


def plot_lrm_dynamics():
    pre = post = np.linspace(-1, 1, 100)
    plt.figure(dpi=200)
    fig, axs = plt.subplots(4, 4)
    for i in range(lrm.shape[0]):
        eta, A, B, C, D = lrm[i, :]
        plt.figure(dpi=200)
        dw = eta * (
                A * (pre[:, None] @ post[None, :]) +
                (B * pre[:, None]) +
                (C * post[None, :]) +
                D
        )
        ax = axs[i // 4, i % 4]
        im = ax.imshow(np.tanh(dw), origin='lower', extent=[-1, 1, -1, 1], vmin=-1, vmax=1)

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(cax=cbar_ax)
    plt.show()
    plt.close()


if __name__ == '__main__':
    plot_lrm_dynamics()
