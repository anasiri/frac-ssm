import torch


def state_passing(
    states, chunk_decay, initial_state=None, out_dtype=None,
):
    """Propagate states between chunks using a float32 recurrence."""
    B, nchunks, R, M = states.shape
    out_dtype = states.dtype if out_dtype is None else out_dtype

    if initial_state is None:
        running = torch.zeros(B, R, M, device=states.device, dtype=torch.float32)
    else:
        running = initial_state.float()

    states_acc = torch.empty(B, nchunks, R, M, device=states.device, dtype=out_dtype)
    for n in range(nchunks):
        states_acc[:, n] = running.to(out_dtype)
        decay = torch.exp(chunk_decay[:, n].float()).unsqueeze(1)
        running = decay * running + states[:, n].float()

    return states_acc, running.float()
