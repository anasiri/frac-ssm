import math

import torch
import torch.nn.functional as F


def chunk_readout(
    u, c, states_acc, W, log_cum_rho, chunk_size,
):
    """Compute chunk readouts in PyTorch.

    Compute decay in float32 and contractions in the input dtype.
    """
    B, L, R = u.shape
    M = c.shape[-1]
    nchunks = math.ceil(L / chunk_size)
    L_padded = nchunks * chunk_size

    if L_padded > L:
        pad_len = L_padded - L
        u = F.pad(u, (0, 0, 0, pad_len), value=0.0)
        c = F.pad(c, (0, 0, 0, pad_len), value=0.0)
        log_cum_rho = F.pad(
            log_cum_rho, (0, 0, 0, pad_len), value=0.0,
        )

    u_chunked = u.view(B, nchunks, chunk_size, R)
    c_chunked = c.view(B, nchunks, chunk_size, M)
    log_cum_rho_chunked = log_cum_rho.view(B, nchunks, chunk_size, M)

    decay = torch.exp(log_cum_rho_chunked.float())
    c_decay = (c_chunked.float() * decay).to(u.dtype)

    h_inter = torch.einsum("bntm,bnrm->bntr", c_decay, states_acc.to(u.dtype))
    h_intra = torch.einsum("bntk,bnkr->bntr", W.to(u.dtype), u_chunked)
    return (h_inter + h_intra).reshape(B, L_padded, R)[:, :L, :]
