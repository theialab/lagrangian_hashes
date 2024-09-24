from arrgh import arrgh

import torch

import laghash._C as laghash_C


def interpolate(coords, codebook, means, stds, feat_begin_idxes, gau_begin_idxes, 
                codebook_bitwidth, resolutions, num_gaus):
    batch, _ = coords.shape
    feature_dim = codebook.shape[-1]
    num_lods = resolutions.shape[0]

    with torch.cuda.amp.autocast(True):
        feats_out, gmm = LagHashGridInterpolate.apply(coords.contiguous(), codebook, means, 
                            stds, feat_begin_idxes, gau_begin_idxes, codebook_bitwidth, 
                            resolutions, num_gaus)
    
    return feats_out, gmm



class LagHashGridInterpolate(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, coords, codebook, means, stds, codebook_first_idxes, gau_first_idxes,
                codebook_bitwidth, resolutions, num_gaus):
        if codebook[0].shape[-1] % 2 == 1:
            raise Exception("The codebook feature dimension needs to be a multiple of 2.")

        assert(coords.shape[-1] in [2, 3])

        if torch.is_autocast_enabled():
            codebook = codebook.half()
            means = means.half()
            stds = stds.half()

        feats, gmm = laghash_C.ops.laghash_interpolate_cuda(
            coords.contiguous(), 
            codebook,
            means,
            stds,
            codebook_first_idxes,
            gau_first_idxes,
            resolutions,
            num_gaus,
            codebook_bitwidth)

        feats = feats.contiguous()
        gmm = gmm.contiguous()
        # arrgh(feats, gmm)
    
        ctx.save_for_backward(coords, codebook, means, stds, codebook_first_idxes, gau_first_idxes)
        ctx.resolutions = resolutions
        ctx.num_lods = len(resolutions)
        ctx.codebook_bitwidth = codebook_bitwidth
        ctx.feature_dim = codebook.shape[-1]
        ctx.num_gaus = num_gaus
        return feats, gmm


    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_feats, grad_gmm):
        coords = ctx.saved_tensors[0]
        codebook = ctx.saved_tensors[1]
        means = ctx.saved_tensors[2]
        stds = ctx.saved_tensors[3]
        codebook_first_idxes = ctx.saved_tensors[4]
        gau_first_idxes = ctx.saved_tensors[5]
        resolutions = ctx.resolutions
        num_gaus = ctx.num_gaus
        codebook_bitwidth = ctx.codebook_bitwidth
        std_requires_grad = ctx.needs_input_grad[3]

        grad_codebook, grad_means, grad_stds = laghash_C.ops.laghash_interpolate_backward_cuda(
                coords.float().contiguous(),
                codebook,
                means,
                stds,
                codebook_first_idxes,
                gau_first_idxes,
                resolutions,
                grad_feats.contiguous(),
                grad_gmm.contiguous(),
                num_gaus,
                codebook_bitwidth,
                std_requires_grad)

        grad_codebook = grad_codebook.contiguous()
        grad_means = grad_means.contiguous()
        grad_stds = grad_stds.contiguous()
        # arrgh(grad_codebook, grad_means, grad_stds)

        if std_requires_grad:
            return (None, grad_codebook, grad_means, grad_stds, None, None, None, None, None)
        else:
            return (None, grad_codebook, grad_means, None, None, None, None, None, None)