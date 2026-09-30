import math

import torch
import torch.nn.functional as F


def chunk_cumsum(
    log_rho: torch.Tensor,  # (B, L, M)
    chunk_size: int,
) -> tuple[
    torch.Tensor,  # (B, L, M)
    torch.Tensor,  # (B, ceil(L / chunk_size), M)
]:
    """Compute within-chunk cumulative log decays in float32."""
    B, L, M = log_rho.shape
    nchunks = math.ceil(L / chunk_size)
    L_padded = nchunks * chunk_size

    x = log_rho.float()
    if L_padded > L:
        x = F.pad(x, (0, 0, 0, L_padded - L))
    x = x.view(B, nchunks, chunk_size, M)

    log_cum_rho = torch.cumsum(x, dim=2)
    chunk_decay = log_cum_rho[:, :, -1].contiguous()
    log_cum_rho = log_cum_rho.reshape(B, L_padded, M)[:, :L, :]
    return log_cum_rho, chunk_decay
