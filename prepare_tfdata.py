import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
from librosa.util import normalize
from scipy.io.wavfile import read
from tqdm.auto import tqdm

tf.config.set_visible_devices([], "GPU")


parser = ArgumentParser()
parser.add_argument("--wav-dir", type=str, default="./LJSpeech-1.1/wavs")
parser.add_argument("--config-file", type=str, default="config.json")
parser.add_argument("--output-dir", type=str, default="tfdata")
parser.add_argument("--num-parts", type=int, default=100)
FLAGS = parser.parse_args()
data_dir = Path(FLAGS.wav_dir)
output_dir = Path(FLAGS.output_dir)
num_parts = FLAGS.num_parts
# load config file
with open(FLAGS.config_file, "r", encoding="utf-8") as f:
    config = json.load(f)
MAX_WAV_VALUE = 32768.0
segment_size = config["segment_size"]
sampling_rate = config["sampling_rate"]
wav_files = sorted(data_dir.glob("*.wav"))
# 90% training, 10% validation
train_files = wav_files[: int(len(wav_files) * 0.9)]
valid_files = wav_files[int(len(wav_files) * 0.9) :]
print(
    f"Found {len(wav_files)} files, "
    f"{len(train_files)} train files, "
    f"{len(valid_files)} validation files"
)


def write_to_tfrecord(files, output_dir: Path, num_parts: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    num_files_per_part = len(files) // num_parts + 1
    for i in tqdm(range(0, len(files), num_files_per_part)):
        j = min(i + num_files_per_part, len(wav_files))
        with tf.io.TFRecordWriter(f"{output_dir}/part_{i:05d}.tfrecord") as file_writer:
            for file_path in wav_files[i:j]:
                sampling_rate, data = read(file_path)
                assert sampling_rate == config["sampling_rate"]
                data = data / MAX_WAV_VALUE
                data = normalize(data) * 0.95
                assert np.min(data) > -1 and np.max(data) < 1
                if data.shape[0] < segment_size:
                    print("Too short, padding...", file_path)
                    data = np.pad(data, [(0, segment_size - data.shape[0])])
                data = tf.constant(data.astype(np.float16))
                file_writer.write(tf.io.serialize_tensor(data).numpy())


# write train files
write_to_tfrecord(train_files, output_dir / "train", num_parts)
# write validation files
write_to_tfrecord(valid_files, output_dir / "val", 1)
print("DONE!")
