"""Fractional chunked scan in PyTorch."""

import torch

from .chunk_cumsum import chunk_cumsum
from .chunk_state import chunk_state
from .state_passing import state_passing
from .chunk_w import chunk_w
from .chunk_readout import chunk_readout


def frac_scan(
    u,
    c,
    kappa,
    rho, # this is rho hat from the paper
    chunk_size,
    initial_state=None,
    return_final_state=False,
):
    """Run the end-to-end fractional chunked scan in PyTorch."""
    assert rho.dtype == torch.float32, "rho must be float32 for numerical stability"
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be float32"

    log_rho = torch.log(rho.clamp(min=1e-6))
    log_cum_rho, chunk_decay = chunk_cumsum(
        log_rho, chunk_size,
    )
    states = chunk_state(
        u, kappa, log_cum_rho, chunk_size,
    )
    states_acc, final_state = state_passing(
        states, chunk_decay, initial_state,
    )
    W = chunk_w(
        c, kappa, log_cum_rho, chunk_size,
    )
    h = chunk_readout(
        u, c, states_acc, W, log_cum_rho, chunk_size,
    )

    if return_final_state:
        return h, final_state
    return h


__all__ = ["frac_scan"]
