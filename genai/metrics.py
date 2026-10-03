"""SSIM / PSNR (used both as losses and as evaluation metrics) and the routing-balance loss."""
import torch
import torch.nn.functional as F


def _gauss_window(size=11, sigma=1.5, channels=3, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[:, None]
    w = (g @ g.t())[None, None]
    return w.expand(channels, 1, size, size).contiguous()


def ssim(x, y, reduction="mean", data_range=1.0):
    """Gaussian-window SSIM (Wang et al., 2004). x, y in [0, data_range], shape B x C x H x W."""
    x, y = x.float(), y.float()
    c = x.shape[1]
    w = _gauss_window(channels=c, device=x.device)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    per_img = m.flatten(1).mean(1)
    return per_img.mean() if reduction == "mean" else per_img


def psnr(x, y, reduction="mean"):
    mse = (x.float() - y.float()).pow(2).flatten(1).mean(1).clamp_min(1e-10)
    p = 10 * torch.log10(1.0 / mse)
    return p.mean() if reduction == "mean" else p


def restoration_loss(pred, target, alpha):
    """L = alpha * L1 + (1 - alpha) * (1 - SSIM)."""
    pred = pred.float()
    return alpha * F.l1_loss(pred, target) + (1 - alpha) * (1 - ssim(pred, target))


def val_score(psnr_val, ssim_val):
    """Validation objective combining reconstruction quality and structure (both ~[0, 1])."""
    return 0.5 * ssim_val + 0.5 * min(psnr_val, 40.0) / 40.0


def balance_loss(w):
    """sum_k (mean_batch w_k - 1/K)^2 from the assignment (computed on balanced batches)."""
    k = w.shape[1]
    return ((w.mean(0) - 1.0 / k) ** 2).sum()
