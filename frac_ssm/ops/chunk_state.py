import math

import torch
import torch.nn.functional as F


def chunk_state(
    u, kappa, log_cum_rho, chunk_size,
    state_fp32=True,
):
    """
    Compute decay in float32 and contractions in the input dtype.

    Args:
        u:           (B, L, R)
        kappa:       (B, L, M)
        log_cum_rho: (B, L, M)  — within-chunk cumsum from chunk_cumsum
        chunk_size:  int

    Returns:
        states: (B, nchunks, R, M), fp32 when state_fp32=True, otherwise u.dtype

    Notes:
        - For unaligned L, log_cum_rho's last valid element is replicated into trailing
          pad positions. This makes lcr[chunk_size-1] of the partial chunk equal lcr[chunk_len-1],
          and routes the per-chunk d_exp.sum to chunk_len-1 for autograd.
    """
    B, L, R = u.shape
    M = kappa.shape[-1]
    nchunks = math.ceil(L / chunk_size)
    L_padded = nchunks * chunk_size

    if L_padded > L:
        u     = F.pad(u,     (0, 0, 0, L_padded - L), value=0.0)
        kappa = F.pad(kappa, (0, 0, 0, L_padded - L), value=0.0)
        last  = log_cum_rho[:, L - 1:L, :]
        log_cum_rho = torch.cat([log_cum_rho, last.expand(-1, L_padded - L, -1)], dim=1)

    u_4d = u.view(B, nchunks, chunk_size, R)
    kappa_4d = kappa.view(B, nchunks, chunk_size, M)
    log_cum_rho_4d = log_cum_rho.float().view(B, nchunks, chunk_size, M)
    lcr_end = log_cum_rho_4d[:, :, -1:, :]
    decay_to_end = torch.exp(lcr_end - log_cum_rho_4d).float()
    scale = (decay_to_end * kappa_4d.float()).to(u.dtype)
    states = torch.einsum('bntr,bntm->bnrm', u_4d, scale)

    states_dtype = torch.float32 if state_fp32 else u.dtype
    return states.to(states_dtype)
