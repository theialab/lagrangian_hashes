"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import os
import math
import numpy as np
import imageio
import time
import trimesh
import tqdm
# from lpips import LPIPS
from arrgh import arrgh
import logging
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from radiance_fields.laghash import LagHashRadianceField
from datasets.tanks_and_temples import TanksTempleDataset
from datasets.nerf_synthetic import SubjectLoader
from examples.utils import TANKS_TEMPLE_SCENES, NERF_SYNTHETIC_SCENES, render_image_with_occgrid, set_random_seed
from nerfacc.estimators.occ_grid import OccGridEstimator

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../config/",
    "config_name": "synthetic_occ.yaml",
}
# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(**_HYDRA_PARAMS)
def run(cfg: DictConfig):
    device = cfg.device
    set_random_seed(42)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg['runtime']['output_dir']
    log.info(f"Saving outputs in: {to_absolute_path(output_path)}")
    writer = SummaryWriter(output_path, purge_step=0)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)

    if cfg.dataset.scene in TANKS_TEMPLE_SCENES:

        # training parameters
        max_steps = cfg.trainer.max_steps
        init_batch_size = cfg.dataset.init_batch_size
        target_sample_batch_size = 1 << 18
        weight_decay = cfg.optimizer.weight_decay
        # dataset parameters
        data_path = os.path.join(cfg.dataset.data_root, cfg.dataset.scene)
        train_dataset = TanksTempleDataset(data_path, split='train', downsample=1,
                                            is_stack=False, num_rays=init_batch_size)
        test_dataset = TanksTempleDataset(data_path, split='test', downsample=1, 
                                          is_stack=True, num_rays=None)
        # scene parameters        
        aabb = train_dataset.scene_bbox.to(device).view(-1)
        near_plane = train_dataset.near_far[0]
        far_plane = train_dataset.near_far[1]
        white_bg = train_dataset.white_bg
        # occupancy parameters
        grid_resolution = cfg.occupancy.grid_resolution
        grid_nlvl = cfg.occupancy.grid_nlvl
        # render parameters
        render_step_size = cfg.render.render_step_size
        alpha_thre = cfg.render.alpha_thre
        cone_angle = cfg.render.cone_angle

    elif cfg.dataset.scene in NERF_SYNTHETIC_SCENES:

        # training parameters
        max_steps = cfg.trainer.max_steps
        init_batch_size = cfg.dataset.init_batch_size
        target_sample_batch_size = 1 << 18
        weight_decay = (
            1e-5 if cfg.dataset.scene in ["materials", "ficus", "drums"] else 1e-6
        )
        # dataset parameters
        train_dataset_kwargs = {}
        test_dataset_kwargs = {}
        train_dataset = SubjectLoader(subject_id=cfg.dataset.scene, root_fp=cfg.dataset.data_root, 
                                      split="train", num_rays=init_batch_size, device=device, 
                                      **train_dataset_kwargs,)
        test_dataset = SubjectLoader(subject_id=cfg.dataset.scene, root_fp=cfg.dataset.data_root, 
                                     split="test", num_rays=None, device=device, **test_dataset_kwargs,)
        # scene parameters
        aabb = torch.tensor(cfg.scene.aabb, device=device)
        near_plane = cfg.scene.near_plane
        far_plane = cfg.scene.far_plane
        # occupancy parameters
        grid_resolution = cfg.occupancy.grid_resolution
        grid_nlvl = cfg.occupancy.grid_nlvl
        # render parameters
        render_step_size = cfg.render.render_step_size
        alpha_thre = cfg.render.alpha_thre
        cone_angle = cfg.render.cone_angle

    else:
        logging.error(f"Invalid scene: {cfg.dataset.scene}")
        assert(0)


    estimator = OccGridEstimator(roi_aabb=aabb, resolution=grid_resolution, 
                                 levels=grid_nlvl).to(device)

    # setup the radiance field we want to train.
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    std_decay_factor = (cfg.model.std_final_factor / cfg.model.std_init_factor) ** (cfg.trainer.size_decay_every/cfg.trainer.max_steps)
    radiance_field = LagHashRadianceField(aabb=estimator.aabbs[-1], 
                                          log2_hashmap_size=cfg.model.log2_hashmap_size, 
                                          num_splashes=cfg.model.num_splashes,
                                          max_resolution=cfg.model.max_resolution, 
                                          std_init_factor=cfg.model.std_init_factor,
                                          fixed_std=cfg.model.fixed_std,
                                          decay_factor=std_decay_factor, 
                                          splits=cfg.model.splits).to(device)
    if cfg.model.load_model_path != "":
        state = torch.load(cfg.model.load_model_path, map_location=device)
        radiance_field.load_state_dict(state['model'])
        estimator.load_state_dict(state['occupancy'])
        checkpoint_steps = state['steps']
        log.info(f"Loaded model from {cfg.model.load_model_path}")
    num_params = sum(p.numel() for p in radiance_field.parameters() if p.requires_grad)
    log.info(f"Number of parameters: {num_params/1e6:.2f}M")
    params_dict = { name : param for name, param in radiance_field.named_parameters()}
    codebook_params = []
    gau_params = []
    rest_params = []
    for name in params_dict:
        if ("means" in name) or ("stds" in name):
            gau_params.append(params_dict[name])
        elif "feats" in name:
            codebook_params.append(params_dict[name])
        else:
            rest_params.append(params_dict[name])
    
    params = []
    gau_lr = cfg.optimizer.learning_rate*cfg.optimizer.gaussian_factor
    params.append({"params": gau_params, "lr": gau_lr,
                   "eps": cfg.optimizer.eps, "weight_decay": 0.0})
    params.append({"params": codebook_params, "lr": cfg.optimizer.learning_rate,
                   "eps": cfg.optimizer.eps, "weight_decay": weight_decay})
    params.append({"params": rest_params, "lr": cfg.optimizer.learning_rate, 
                   "eps": cfg.optimizer.eps, "weight_decay": weight_decay}) # 
    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=100
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(m*max_steps) for m in cfg.scheduler.milestones],
                gamma=cfg.scheduler.gamma,
            ),
        ]
    )
    # lpips_net = LPIPS(net="vgg").to(device)
    # lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    # lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

    # training
    tic = time.time()
    for step in range(max_steps + 1):
        radiance_field.train()
        estimator.train()

        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * render_step_size

        # update occupancy grid
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        # render
        rgb, acc, depth, kl_div, n_rendering_samples, mip_loss = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        if n_rendering_samples == 0:
            continue

        if target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)

        # compute loss
        rgb_loss = F.smooth_l1_loss(rgb, pixels)
        loss_warm_up = min(4*step/max_steps, math.exp(-4*step/max_steps))
        mip_loss = mip_loss.mean()
        sigma_loss, surf_loss, i = 0, 0, 0
        for idx in range(radiance_field.n_levels):
            resolution = radiance_field.mlp_base.encoding.resolutions[idx]
            stds = radiance_field.mlp_base.encoding.get_stds(idx)
            if stds is not None:
                sigma_diff = F.relu(stds - 2/resolution)
                lod_sigma_loss = torch.mean(sigma_diff)
                sigma_loss += lod_sigma_loss
                i += 1
        if i:
            sigma_loss /= i
            surf_loss = kl_div.mean()

        loss = rgb_loss
        if cfg.trainer.weight_surface:
            loss += cfg.trainer.weight_surface * loss_warm_up * surf_loss
        if cfg.trainer.weight_sigma and (not cfg.model.fixed_std):
            loss += cfg.trainer.weight_sigma * loss_warm_up * sigma_loss
        if cfg.trainer.weight_mip:
            loss += cfg.trainer.weight_mip * mip_loss

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        if (step % cfg.trainer.size_decay_every == cfg.trainer.size_decay_every-1) and cfg.model.fixed_std:
            radiance_field.mlp_base.encoding.update_factor()

        # if step % cfg.trainer.log_every == 0:
        #     writer.add_scalar("train/rgb_loss", rgb_loss, step)
        #     writer.add_scalar("train/surface_loss", surf_loss, step)
        #     writer.add_scalar("train/sigma_loss", sigma_loss, step)
        #     writer.add_scalar("train/mip_loss", mip_loss, step)

        if step % cfg.trainer.save_every == 0:
            state_dict = {
                "steps": step,
                "model": radiance_field.state_dict(),
                "occupancy": estimator.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state_dict, f"{output_path}/model.pth")

            for idx in range(radiance_field.n_levels):
                means = radiance_field.mlp_base.encoding.get_means(idx)
                if means is not None:
                    means = means.reshape(-1, means.shape[-1])

                    means_cloud = trimesh.PointCloud(means.cpu().detach().numpy())
                    if step:
                        os.remove(os.path.join(output_path, f'means_lod{idx}@{step-cfg.trainer.save_every:05d}.ply'))
                    means_cloud.export(os.path.join(output_path, f'means_lod{idx}@{step:05d}.ply')) # 

        if step % cfg.trainer.visualize_every == 0:
            if step:
                # evaluation
                radiance_field.eval()
                estimator.eval()

                with torch.no_grad():
                    data = test_dataset[0]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]

                    rgb, acc, depth, kl_div, _, mip_loss = render_image_with_occgrid(
                        radiance_field,
                        estimator,
                        rays,
                        # rendering options
                        near_plane=near_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=cone_angle,
                        alpha_thre=alpha_thre,
                    )
                    visualize = torch.concatenate([rgb, pixels], dim=1)
                    writer.add_image("visual/rgb", visualize,  step, dataformats="HWC")
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(loss) / np.log(10.0)
            log.info(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"psnr={psnr:.2f} | loss={rgb_loss:.5f} | "
                f"surf_loss={surf_loss:.5f} | "
                f"sigma_loss={sigma_loss:.5f} | "
                f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                f"max_depth={depth.max():.3f} | "
            )


    # evaluation
    radiance_field.eval()
    estimator.eval()

    psnrs = []
    # lpips = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_dataset))):
            data = test_dataset[i]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            rgb, acc, depth, kl_div, _, mip_loss = render_image_with_occgrid(
                radiance_field,
                estimator,
                rays,
                # rendering options
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )
            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())
            # lpips.append(lpips_fn(rgb, pixels).item())
            imageio.imwrite(
                f"{output_path}/test/rgb_test_{i}.png",
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )
            # imageio.imwrite(
            #     f"{output_path}/rgb_error_{i}.png",
            #     (
            #         (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
            #     ).astype(np.uint8),
            # )
    psnr_avg = sum(psnrs) / len(psnrs)
    # lpips_avg = sum(lpips) / len(lpips)
    logging.info(f"evaluation: psnr_avg={psnr_avg}") # , lpips_avg={lpips_avg}
    writer.add_scalar("test/psnr", psnr_avg, max_steps)
    # writer.add_scalar("test/lpips", lpips_avg, max_steps)
    with open(f"{output_path}/metrics.txt", "w") as fp:
        fp.write(f"PSNR:{psnr_avg:.3f}") # , LPIPS:{lpips_avg:.3f}
    writer.close()


if __name__ == "__main__":
    run()
