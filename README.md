# Simple HiFiGAN

Another HiFiGAN implementation using Pytorch.


```bash
pip install -U pip
pip install -r requirements.txt
```

### Prepare data

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -O - | tar -xj
python prepare_tfdata.py --wav-dir ./LJSpeech-1.1/wavs --config-file config.json --output-dir tfdata --num-parts 100
```

### Train model

```bash
python train.py
```

### Credit
- We reuse most of the model code from HiFiGAN official repo: https://github.com/jik876/hifi-gan
