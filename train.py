import torch  # isort:skip
import itertools
import json
import time
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from model import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from tfloader import get_util_funcs

tf.config.set_visible_devices([], "GPU")

parser = ArgumentParser()
parser.add_argument("--config-file", type=Path, default="config.json")
parser.add_argument("--log-dir", type=Path, default="logs")
parser.add_argument("--tfdata-dir", type=Path, default="tfdata")
parser.add_argument("--ckpt-dir", type=Path, default="ckpts")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--ckpt-interval", type=int, default=5000)

FLAGS = parser.parse_args()

config_file = FLAGS.config_file
log_dir = FLAGS.log_dir
tfdata_dir = FLAGS.tfdata_dir
ckpt_dir = FLAGS.ckpt_dir
ckpt_interval = FLAGS.ckpt_interval

# credit: https://github.com/karpathy/nanoGPT/blob/master/train.py#L72-L112
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed(FLAGS.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = FLAGS.device
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = FLAGS.compile
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


with open(config_file, "r", encoding="utf-8") as f:
    h = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
generator = Generator(h).to(device)
mpd = MultiPeriodDiscriminator().to(device)
msd = MultiScaleDiscriminator().to(device)
if compile:
    generator = torch.compile(generator)
    mpd = torch.compile(mpd)
    msd = torch.compile(msd)

optim_g = torch.optim.AdamW(
    generator.parameters(),
    h.learning_rate,
    betas=[h.adam_b1, h.adam_b2],
    fused=True,
)
optim_d = torch.optim.AdamW(
    itertools.chain(msd.parameters(), mpd.parameters()),
    h.learning_rate,
    betas=[h.adam_b1, h.adam_b2],
    fused=True,
)
g_scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
d_scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

generator.train()
mpd.train()
msd.train()
train_writer = SummaryWriter(log_dir / "train", flush_secs=30)
val_writer = SummaryWriter(log_dir / "val", flush_secs=30)
mel_spectrogram, mel_spectrogram_loss, get_data_iter = get_util_funcs(
    config_file, device
)
train_dataloader = get_data_iter(tfdata_dir / "train", h.batch_size)
val_dataloader = get_data_iter(tfdata_dir / "val", h.batch_size)
val_data_iter = iter(val_dataloader)

for i, batch in tqdm(enumerate(train_dataloader)):
    start_b = time.time()
    y = batch["wav"]
    x = batch["mel"]
    y_mel = batch["mel_loss"]
    y = y.unsqueeze(1)

    optim_d.zero_grad(set_to_none=True)
    with ctx:
        y_g_hat = generator(x)
        y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
    y_g_hat_mel = mel_spectrogram_loss(y_g_hat.squeeze(1))
    # MPD
    loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
        y_df_hat_r, y_df_hat_g
    )
    # MSD
    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
        y_ds_hat_r, y_ds_hat_g
    )
    loss_disc_all = loss_disc_s + loss_disc_f

    d_scaler.scale(loss_disc_all).backward()
    d_scaler.step(optim_d)
    d_scaler.update()

    # Generator
    optim_g.zero_grad(set_to_none=True)

    with ctx:
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
    # L1 Mel-Spectrogram Loss
    loss_mel = F.l1_loss(y_mel, y_g_hat_mel)
    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
    loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * 45

    g_scaler.scale(loss_gen_all).backward()
    g_scaler.step(optim_g)
    g_scaler.update()
    train_writer.add_scalar("loss_disc_f", loss_disc_f, global_step=i)
    train_writer.add_scalar("loss_disc_s", loss_disc_s, global_step=i)
    train_writer.add_scalar("loss_disc_all", loss_disc_all, global_step=i)
    train_writer.add_scalar("gen_loss_total", loss_gen_all, global_step=i)
    train_writer.add_scalar("loss_mel", loss_mel, global_step=i)
    train_writer.add_scalar("loss_fm_f", loss_fm_f.float(), global_step=i)
    train_writer.add_scalar("loss_fm_s", loss_fm_s.float(), global_step=i)
    train_writer.add_scalar("loss_gen_f", loss_gen_f, global_step=i)
    train_writer.add_scalar("loss_gen_s", loss_gen_s, global_step=i)
    train_writer.add_scalar("d_grad_scale", d_scaler.get_scale(), global_step=i)
    train_writer.add_scalar("g_grad_scale", g_scaler.get_scale(), global_step=i)

    if i % 10 == 0:
        # validation
        with torch.no_grad():
            generator.eval()
            batch = next(val_data_iter)
            y = batch["wav"]
            x = batch["mel"]
            y_mel = batch["mel_loss"]
            y = y.unsqueeze(1)

            with ctx:
                y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram_loss(y_g_hat.squeeze(1))
            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel)
            val_writer.add_scalar("loss_mel", loss_mel, global_step=i)
            generator.train()

    if i % ckpt_interval == 0:
        torch.save(
            {
                "generator": generator.state_dict(),
                "mpd": mpd.state_dict(),
                "msd": msd.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "g_scaler": g_scaler.state_dict(),
                "d_scaler": d_scaler.state_dict(),
            },
            ckpt_dir / f"ckpt_{i:07d}.pt",
        )
