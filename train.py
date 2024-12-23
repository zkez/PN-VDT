import torch
import os
import math
import argparse
import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models import get_models
from datasets import get_dataset
from diffusion import create_diffusion
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, setup_distributed,
                   get_experiment_dir)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # 设置分布式环境
    setup_distributed()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # 创建实验文件夹（只在 rank=0 主进程进行）
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # 创建模型
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    args.latent_size = args.image_size // 8
    model = get_models(args)
    diffusion = create_diffusion(timestep_respacing="1000")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)

    # 创建并初始化 EMA
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    # 将模型封装为 DDP
    model = DDP(model.to(device), device_ids=[local_rank])

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # 冻结 VAE
    vae.requires_grad_(False)

    # 准备数据
    train_dataset, val_dataset = get_dataset(args)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} videos ({args.data_path})")

    # 学习率调度器
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # 初始化 EMA
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()
    num_update_steps_per_epoch = math.ceil(len(loader))
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 训练循环
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(loader):
            x, label = video_data[0].to(device, non_blocking=True), video_data[1].to(device)
            b, _, _, _, _ = x.shape

            # VAE 编码：将视频帧映射到 VAE 的潜空间
            with torch.no_grad():
                x = rearrange(x, 'b f c h w -> (b f) c h w')
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, '(b f) c h w -> b f c h w', b=b)

            # 构造条件
            model_kwargs = dict(y=None)  # 这里没有添加标签或文本等条件

            # 随机选择扩散时刻并计算 Diffusion loss
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
            loss.backward()

            # 梯度裁剪
            if train_steps < args.start_clip_iter:
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

            # 优化器 & EMA
            lr_scheduler.step()
            if train_steps % args.gradient_accumulation_steps == 0 and train_steps > 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # 日志记录
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # 计算全局平均 loss
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) "
                            f"Train Loss: {avg_loss:.4f}, "
                            f"Gradient Norm: {gradient_norm:.4f}, "
                            f"Steps/Sec: {steps_per_sec:.2f}")

                # 重置计数
                running_loss = 0
                log_steps = 0
                start_time = time()

            # 定期保存 checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train Latte with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/nlst_train.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
