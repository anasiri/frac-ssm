import torch
import torch.nn.functional as F


def frac_compute_scan_params(
    model,
    x_conv,
    x,
    attention_mask=None,
):
    """Compute scan parameters from projected features in PyTorch."""
    dyn = model.dyn_proj(x).float()
    dt_logits, alpha_logits, lambda_logits = dyn.split(
        model.num_heads, dim=-1,
    )
    dt_bias = model.dt_bias.float()
    dt = F.softplus(dt_logits + dt_bias)
    dt = dt.clamp(
        min=model.config.frac_dt_min, max=model.config.frac_dt_max,
    )
    alpha = torch.sigmoid(alpha_logits)
    alpha = model.config.frac_alpha_min + alpha * (
        model.config.frac_alpha_max - model.config.frac_alpha_min
    )
    lam = torch.sigmoid(lambda_logits)
    lam = model.config.frac_lambda_min + lam * (
        model.config.frac_lambda_max - model.config.frac_lambda_min
    )

    rw = model.rw_proj(x_conv).float()
    rw = rw.view(
        x_conv.size(0),
        x_conv.size(1),
        2,
        model.num_heads,
        model.num_modes,
    )
    read_residual = rw[:, :, 0]
    write_residual = rw[:, :, 1]

    log_tau = model.base_log_tau.float()[None, None, None, :]
    prior = -alpha[..., None] * log_tau

    read_gate = torch.sigmoid(model.read_residual_logit.float())
    read_logits = prior + read_gate * read_residual
    read_weights = torch.softmax(read_logits / model.read_temp, dim=-1)

    write_gate = torch.sigmoid(model.write_residual_scale.float())
    write_logits = prior + write_gate * write_residual
    write_weights = torch.softmax(write_logits, dim=-1)

    tau = torch.exp(log_tau)
    lambda_factor = torch.pow(
        lam.clamp_min(1e-4), 1.0 / alpha.clamp_min(1e-4)
    )
    inv_tau_eff = lambda_factor[..., None] / tau.clamp_min(1e-6)
    rho = torch.exp(-dt[..., None] * inv_tau_eff)
    rho = rho.clamp(min=1e-6, max=1.0 - 1e-6)
    beta = 1.0 - rho

    bsz, seqlen, _ = x_conv.shape
    u = x_conv.view(
        bsz, seqlen, model.num_heads, model.head_dim,
    )
    u_flat = u.permute(0, 2, 1, 3).reshape(
        bsz * model.num_heads, seqlen, model.head_dim,
    )
    c_flat = read_weights.permute(0, 2, 1, 3).reshape(
        bsz * model.num_heads, seqlen, model.num_modes,
    )
    kappa_flat = beta.mul(write_weights).permute(0, 2, 1, 3).reshape(
        bsz * model.num_heads, seqlen, model.num_modes,
    )
    rho_flat = rho.permute(0, 2, 1, 3).reshape(
        bsz * model.num_heads, seqlen, model.num_modes,
    )

    if attention_mask is not None:
        mask_flat = attention_mask[:, None, :].expand(
            bsz, model.num_heads, seqlen,
        ).reshape(
            bsz * model.num_heads, seqlen, 1,
        ).to(kappa_flat.dtype)
        kappa_flat = kappa_flat * mask_flat
        rho_flat = rho_flat * mask_flat + (1.0 - mask_flat)

    return u_flat, c_flat, kappa_flat, rho_flat
