MIN_PROBABILITY = 1e-30
LOG_ZERO = -69


def multinomial_kl(log_prob1, log_prob2):  # compute KL loss on log_prob
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def extract(a, t, x_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_onehot_to_index(log_x):
    """argmax(log_x, dim=1)"""
    return log_x.argmax(1)


def split_integer(total: int, n: int) -> list[int]:
    """
    Split `total` into `n` non-negative integers that add up to `total`,
    differing by at most 1.
    """
    q, r = divmod(total, n)
    # First r parts get (q+1), the rest get q
    return [q + 1 if i < r else q for i in range(n)]
