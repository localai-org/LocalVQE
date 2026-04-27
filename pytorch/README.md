# LocalVQE — PyTorch Reference

This directory contains the PyTorch definition of LocalVQE. It is provided
for research, verification, and downstream fine-tuning. For production /
end-user inference, use the GGML build in [`../ggml/`](../ggml/).

## Install

```bash
pip install -r requirements.txt
```

(Or use `uv`, `poetry`, etc. — only `torch` and `pyyaml` are required.)

## Use

```python
import yaml, torch
from localvqe.model import LocalVQE

cfg = yaml.safe_load(open("configs/default.yaml"))
model = LocalVQE(
    **cfg["model"],
    n_freqs=cfg["audio"]["n_freqs"],
)
model.eval()

# Load published checkpoint (torch format, from Hugging Face)
state = torch.load("best.pt", map_location="cpu")
model.load_state_dict(state["model"])

# Bake the trained AlignBlock softmax temperature (carried in the
# checkpoint as a buffer) into the smoothing-conv weights. The GGML
# graph has no temperature parameter, so PyTorch inference must call
# this to match its behavior; without it the model runs at the default
# 1.0 and loses several dB of FE-ST ERLE on real recordings.
model.align.fold_temperature()

# Process a (mic, far) pair — shape (B, N_samples)
with torch.no_grad():
    enhanced = model(mic_pcm, far_pcm)
```

## Files

```
localvqe/
  model.py    LocalVQE + DCTEncoder + DCTDecoder
  blocks.py   FE, EncoderBlock, DecoderBlock, S4DBottleneck
  align.py    AlignBlock (soft delay cross-attention)
  ccm.py      Complex convolving mask (real-valued arithmetic)
configs/
  default.yaml   Architecture + audio settings for the published model
```

Only the model definition is included here; no training, loss, or data
pipeline code ships in this repository.
