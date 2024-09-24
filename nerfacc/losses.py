import torch
from torch import Tensor

from .scan import inclusive_sum
from .volrend import accumulate_along_rays


def distortion(
    weights: Tensor,
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Tensor,
    n_rays: int,
) -> Tensor:
    """Distortion Regularization proposed in Mip-NeRF 360.

    Args:
        weights: The flattened weights of the samples. Shape (n_samples,)
        t_starts: The start points of the samples. Shape (n_samples,)
        t_ends: The end points of the samples. Shape (n_samples,)
        ray_indices: The ray indices of the samples. LongTensor with shape (n_samples,)
        n_rays: The total number of rays.

    Returns:
        The per-ray distortion loss with the shape (n_rays, 1).
    """
    assert (
        weights.shape == t_starts.shape == t_ends.shape == ray_indices.shape
    ), (
        f"the shape of the inputs are not the same: "
        f"weights {weights.shape}, t_starts {t_starts.shape}, "
        f"t_ends {t_ends.shape}, ray_indices {ray_indices.shape}"
    )
    t_mids = 0.5 * (t_starts + t_ends)
    t_deltas = t_ends - t_starts
    loss_uni = (1 / 3) * (t_deltas * weights.pow(2))
    loss_bi_0 = weights * t_mids * inclusive_sum(weights, indices=ray_indices)
    loss_bi_1 = weights * inclusive_sum(weights * t_mids, indices=ray_indices)
    loss_bi = 2 * (loss_bi_0 - loss_bi_1)
    loss = loss_uni + loss_bi
    loss = accumulate_along_rays(loss, None, ray_indices, n_rays)
    return loss

####

def ray_samples_to_sdist(ray_samples):
    """Convert ray samples to s space"""
    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)  # (num_rays, num_samples + 1)
    return sdist


def lossfun_distortion(t, w):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def distortion_loss(weights, t_starts, t_ends):
    """From mipnerf360"""
    # takes samples and weights as proposed by the 'nerf' network
    # c = ray_samples_to_sdist(ray_samples_list[-1])
    c = torch.cat([t_starts[..., ], t_ends[..., -1:]], dim=-1)
    w = weights
    loss = lossfun_distortion(c, w).view(-1, 1) #torch.mean(
    return loss

####

# class EffDistLoss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, w, m, interval):
#         '''
#         Efficient O(N) realization of distortion loss.
#         There are B rays each with N sampled points.
#         w:        Float tensor in shape [B,N]. Volume rendering weights of each point.
#         m:        Float tensor in shape [B,N]. Midpoint distance to camera of each point.
#         interval: Scalar or float tensor in shape [B,N]. The query interval of each point.
#         '''
#         n_rays = np.prod(w.shape[:-1])
#         wm = (w * m)
#         w_cumsum = w.cumsum(dim=-1)
#         wm_cumsum = wm.cumsum(dim=-1)

#         w_total = w_cumsum[..., [-1]]
#         wm_total = wm_cumsum[..., [-1]]
#         w_prefix = torch.cat([torch.zeros_like(w_total), w_cumsum[..., :-1]], dim=-1)
#         wm_prefix = torch.cat([torch.zeros_like(wm_total), wm_cumsum[..., :-1]], dim=-1)
#         loss_uni = (1/3) * interval * w.pow(2)
#         loss_bi = 2 * w * (m * w_prefix - wm_prefix)
#         if torch.is_tensor(interval):
#             ctx.save_for_backward(w, m, wm, w_prefix, w_total, wm_prefix, wm_total, interval)
#             ctx.interval = None
#         else:
#             ctx.save_for_backward(w, m, wm, w_prefix, w_total, wm_prefix, wm_total)
#             ctx.interval = interval
#         ctx.n_rays = n_rays
#         return (loss_bi.sum() + loss_uni.sum()) / n_rays

#     @staticmethod
#     @torch.autograd.function.once_differentiable
#     def backward(ctx, grad_back):
#         interval = ctx.interval
#         n_rays = ctx.n_rays
#         if interval is None:
#             w, m, wm, w_prefix, w_total, wm_prefix, wm_total, interval = ctx.saved_tensors
#         else:
#             w, m, wm, w_prefix, w_total, wm_prefix, wm_total = ctx.saved_tensors
#         grad_uni = (1/3) * interval * 2 * w
#         w_suffix = w_total - (w_prefix + w)
#         wm_suffix = wm_total - (wm_prefix + wm)
#         grad_bi = 2 * (m * (w_prefix - w_suffix) + (wm_suffix - wm_prefix))
#         grad = grad_back * (grad_bi + grad_uni) / n_rays
#         return grad, None, None, None

# eff_distloss = EffDistLoss.apply