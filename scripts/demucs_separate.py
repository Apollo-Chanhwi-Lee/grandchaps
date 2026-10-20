"""Vocals of a take with htdemucs: separate.py <in.wav> <out vocals.wav>  (audio read and written with soundfile)."""
import sys
import numpy as np, soundfile as sf, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
src, dst = sys.argv[1:3]
y, sr = sf.read(src, dtype="float32", always_2d=True)
model = get_model("htdemucs"); model.eval()
x = torch.from_numpy(y.T)
if x.shape[0] == 1: x = x.repeat(2, 1)
if sr != model.samplerate:
    import librosa
    x = torch.from_numpy(np.stack([librosa.resample(c.numpy(), orig_sr=sr, target_sr=model.samplerate) for c in x]))
ref = x.mean(0); x = (x - ref.mean()) / (ref.std() + 1e-8)
with torch.no_grad():
    out = apply_model(model, x[None], device="cpu", shifts=1, split=True, overlap=0.25)[0]
voc = out[model.sources.index("vocals")] * (ref.std() + 1e-8) + ref.mean()
sf.write(dst, voc.T.numpy(), model.samplerate)
print("vocals", dst, voc.shape[1] / model.samplerate, "s")
