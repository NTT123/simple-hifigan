import json

import numpy as np
import tensorflow as tf
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def get_util_funcs(config_file, device):
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    segment_size = config["segment_size"]
    sampling_rate = config["sampling_rate"]
    n_fft = config["n_fft"]
    num_mels = config["num_mels"]
    fmin = config["fmin"]
    fmax = config["fmax"]
    fmax_loss = config["fmax_for_loss"]
    win_size = config["win_size"]
    hop_size = config["hop_size"]

    mel = librosa_mel_fn(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
    )
    mel_basis = torch.from_numpy(mel).float().to(device)
    hann_window = torch.hann_window(win_size).to(device)
    mel_loss = librosa_mel_fn(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax_loss
    )
    mel_basis_loss = torch.from_numpy(mel_loss).float().to(device)

    def mel_spectrogram(y, center=False):
        y = torch.nn.functional.pad(
            y,
            (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
            mode="reflect",
        )
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(mel_basis, spec)
        spec = spectral_normalize_torch(spec)
        return spec

    def mel_spectrogram_loss(y, center=False):
        y = torch.nn.functional.pad(
            y,
            (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
            mode="reflect",
        )
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(mel_basis_loss, spec)
        spec = spectral_normalize_torch(spec)
        return spec

    def sample_slice(x):
        x = tf.io.parse_tensor(x, out_type=tf.float16)
        x = tf.reshape(x, (-1,))
        start_idx = tf.random.uniform(
            shape=(), maxval=tf.shape(x)[0] - segment_size + 1, dtype=tf.int32
        )
        return x[start_idx : start_idx + segment_size]

    def get_data_iter(data_dir, batch_size):
        files = tf.data.Dataset.list_files(f"{data_dir}/part_*.tfrecord")
        files = files.repeat().shuffle(len(files))
        ds = (
            tf.data.TFRecordDataset(files, num_parallel_reads=4)
            .map(sample_slice, num_parallel_calls=4)
            .shuffle(batch_size * 10)
            .batch(batch_size)
            .prefetch(1)
        )
        for batch in ds.as_numpy_iterator():
            with torch.no_grad():
                wav = torch.from_numpy(batch).to(
                    device, dtype=torch.float32, non_blocking=True
                )
                mel = mel_spectrogram(wav)
                mel_loss = mel_spectrogram_loss(wav)
            yield {"wav": wav, "mel": mel, "mel_loss": mel_loss}

    return mel_spectrogram, mel_spectrogram_loss, get_data_iter
