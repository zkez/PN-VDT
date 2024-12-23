import os
import torch
import random
from tqdm import tqdm
from torch.nn import functional as F
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from os.path import join as j
import argparse
import lpips
from omegaconf import OmegaConf

import sys
sys.path.append('./')
from models.conditional_vae import AutoencoderKL
from datasets import get_dataset
from utils import compute_3frame_perceptual_loss_2d


def main(args):
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取数据加载器
    train_dataset, val_dataset = get_dataset(args)

    # 初始化模型
    up_and_down = args.up_and_down
    attention_levels = (False,) * len(up_and_down)
    vae = AutoencoderKL(
        spatial_dims=3,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        num_channels=args.up_and_down,
        latent_channels=3,
        num_res_blocks=args.num_res_layers,
        attention_levels=attention_levels,
        norm_num_groups=1,  
    )
    if len(args.vae_path):
        vae.load_state_dict(torch.load(args.vae_path, map_location=device))

    discriminator = PatchDiscriminator(
        spatial_dims=3, 
        num_channels=256,
        in_channels=args.out_channels, 
        out_channels=1
    )
    if len(args.dis_path):
        discriminator.load_state_dict(torch.load(args.dis_path, map_location=device))

    lpips_loss = lpips.LPIPS(net='alex').to(device)

    # 模型到设备
    vae = vae.to(device)
    discriminator = discriminator.to(device)

    vae.requires_grad_(False).eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    perceptual_weight = 0.001
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

    val_interval = args.val_inter
    save_interval = args.save_inter
    autoencoder_warm_up_n_epochs = (
        0 
        if (len(args.vae_path) and len(args.dis_path)) 
        else args.autoencoder_warm_up_n_epochs
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataset), desc=f"Epoch {epoch+1}")

        discriminator.train()
        for step, batch in enumerate(train_dataset):
            video, label = batch[1].to(device), batch[2].to(device)

            reconstruction, _, _ = vae(video)

            # 1) 重建损失 (标量)
            recons_loss = F.mse_loss(reconstruction.float(), video.float())

            # 2) 感知损失 (LPIPS)
            p_loss = compute_3frame_perceptual_loss_2d(
                reconstruction.float(), 
                video.float(), 
                perceptual_loss_func=lpips_loss
            )
            if p_loss.dim() > 0:
                p_loss = p_loss.mean()

            # 生成器部分 (初始只是重建+感知)
            loss_g = recons_loss + perceptual_weight * p_loss

            # 如超过 warm-up 期，再加对抗损失
            if epoch + 1 > autoencoder_warm_up_n_epochs:
                # 判别器对假图 => logits_fake
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                # 假图生成器损失 => 需要标量
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                # 如 generator_loss 可能返回 [N], 做 mean => 标量
                if generator_loss.dim() > 0:
                    generator_loss = generator_loss.mean()

                loss_g = loss_g + adv_weight * generator_loss

            # =========== 反向传播生成器 (VAE) ===========
            # 先清理过往梯度 (如要训练VAE的话需 vae.requires_grad_(True))
            # 由于此处是你写的 'vae.requires_grad_(False)', 就不会更新 VAE，除非你想改
            loss_g.backward()  # loss_g 是标量 => OK

            # =========== 训练判别器 ===========
            # 仅在超过 warm-up 后
            if epoch + 1 > autoencoder_warm_up_n_epochs:
                optimizer_d.zero_grad(set_to_none=True)
                # 判别器对 假图
                logits_fake = discriminator(reconstruction.detach().float())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                if loss_d_fake.dim() > 0:
                    loss_d_fake = loss_d_fake.mean()

                # 判别器对 真图
                logits_real = discriminator(video.detach().float())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                if loss_d_real.dim() > 0:
                    loss_d_real = loss_d_real.mean()

                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                # 这里做完再乘权重 adv_weight
                loss_d = adv_weight * discriminator_loss
                loss_d.backward()  # 标量 => OK
                optimizer_d.step()
            else:
                loss_d = torch.tensor(0.0, device=device)
                generator_loss = torch.tensor(0.0, device=device)

            progress_bar.update(1)
            logs = {
                "gen_loss": loss_g.detach().item(), 
                "dis_loss": loss_d.detach().item(),
                "pp_loss": p_loss.detach().item(),
                "adv_loss": generator_loss.detach().item(),
            }
            progress_bar.set_postfix(**logs)
            global_step += 1

        progress_bar.close()

        # ========== 验证 / 保存模型 ==========
        if (epoch + 1) % val_interval == 0 or epoch == args.num_epochs - 1:
            total_mse_loss = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataset):
                    video, label = batch[1].to(device), batch[2].to(device)
                    val_recon, _, _ = vae(video)
                    mse_loss = F.mse_loss(val_recon, video)
                    total_mse_loss += mse_loss.item()
            average_mse_loss = total_mse_loss / len(val_dataset)
            print(f'Epoch {epoch + 1}, Average MSE Loss: {average_mse_loss:.4f}')

        if (epoch + 1) % save_interval == 0 or epoch == args.num_epochs - 1:
            gen_path = os.path.join(args.project_dir, 'gen_save')
            dis_path = os.path.join(args.project_dir, 'dis_save')
            os.makedirs(gen_path, exist_ok=True)
            os.makedirs(dis_path, exist_ok=True)
            torch.save(vae.state_dict(), os.path.join(gen_path, 'vae.pth'))
            torch.save(discriminator.state_dict(), os.path.join(dis_path, 'dis.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/nlst_train_vae.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
