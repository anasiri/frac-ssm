def frac_skip_mix(
    model,
    mixed_flat,
    x_conv,
    out_dtype=None,
):
    """Apply the output skip connection and reshape in PyTorch."""
    if out_dtype is None:
        out_dtype = x_conv.dtype

    bsz, seqlen, _ = x_conv.shape
    u = x_conv.view(bsz, seqlen, model.num_heads, model.head_dim)
    mixed = mixed_flat.view(
        bsz, model.num_heads, seqlen, model.head_dim,
    ).permute(0, 2, 1, 3)
    mixed = mixed + u.to(mixed.dtype) * model.D[None, None].to(mixed.dtype)
    mixed = mixed.reshape(bsz, seqlen, model.inner_dim)
    return mixed.to(out_dtype)
