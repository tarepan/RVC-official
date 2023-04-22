"""
Load preprocessed 16kHz audios, then convert them to HuBERT feature series.

[Before Run]
    {exp_dir}/
        extract_f0_feature.log  # Not required, but can exist
        1_16k_wavs/
            {xxx}.wav           # input, preprocessed 16kHz audio

[After Run]
    {exp_dir}/
        extract_f0_feature.log  # Appended
        1_16k_wavs/
            {xxx}.wav           # No change
        3_feature256/
            {xxx}.npy           # Output, generated HuBERT feature series
"""

import os, sys, traceback

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils
import soundfile as sf


def readwave(wav_path: str, normalize=False):
    """Read a audio from the path and normalize if configured.

    Args:
        wav_path  - Path to .wav file, which should contain sr=16000 audio
        normalize - Whether to normalize audio
    Returns:
        :: (B=1, T) - Audio waveform
    """
    # Load :: mono (T,) | stereo (T, Channel=2)
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()

    # Stereo-to-Mono :: (T,) | (T, Channel=2) -> (T,)
    if feats.dim() == 2:
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()

    # Normalize :: (T,) -> (T,)
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    
    # Reshape :: (T,) -> (B=1, T)
    feats = feats.view(1, -1)

    return feats


# ⚡Configs
## args
n_part: int =  int(sys.argv[2]) # The number of GPUs
i_part: int =  int(sys.argv[3]) # GPU index
if len(sys.argv) == 5:
    exp_dir: str = sys.argv[4]  # Experiment directory
else:
    i_gpu:   str = sys.argv[4]  # GPU specifier
    exp_dir: str = sys.argv[5]  # Experiment directory
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
## Model
model_path = "hubert_base.pt"
## I/O
wavPath = f"{exp_dir}/1_16k_wavs"
outPath = f"{exp_dir}/3_feature256"
os.makedirs(outPath, exist_ok=True)
## Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚡Logger
f = open(f"{exp_dir}/extract_f0_feature.log", "a+")
def printt(strr):
    """Print to STDOUT and log file."""
    print(strr)
    f.write(f"{strr}\n")
    f.flush()

printt(sys.argv)
printt(exp_dir)

# Initialization - HuBERT Wave-to-Unit model
## Accept only 16kHz audio, yield hop_size 320 (20msec/frame) unit series
printt(f"load model on {device} from {model_path}")
models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix="")
model = models[0].to(device)
if device != "cpu":
    model = model.half()
model.eval()

#⚡ Data selection - In charge on this GPU (i, i+n*1, i+n*2, ...)
todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]

# Run
report_interval = max(1, len(todo) // 10)
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt(f"all-feature-{len(todo)}")
    for idx, file in enumerate(todo):
        # Extract unit series from audio file, then save it as a .npy file.
        try:
            if file.endswith(".wav"):
                #⚡ Path and Validation
                wav_path = f"{wavPath}/{file}"                       # e.g. f"{exp_dir}/1_16k_wavs/{xxx}.wav"
                out_path = f"{outPath}/{file.replace('wav', 'npy')}" # e.g. f"{exp_dir}/3_feature256/{xxx}.npy"
                if os.path.exists(out_path):
                    continue

                # Forward
                ## wave :: (B=1, T) - Audio waveform
                wave = readwave(wav_path, normalize=saved_cfg.task.normalize)
                ## mask placeholder :: (B=1, T) -> (B=1, T)
                mask_placeholder = torch.BoolTensor(wave.shape).fill_(False).to(device)
                ## Wave-to-Unit :: (B=1, T) -> (B=1, Frame, Feat) -> (Frame, Feat) - Unit series
                inputs = {
                    "source": wave.half().to(device) if device != "cpu" else wave.to(device),
                    "padding_mask": mask_placeholder,
                    "output_layer": 9,
                }
                with torch.no_grad():
                    feats = model.final_proj(model.extract_features(**inputs)[0])
                feats = feats.squeeze(0).float().cpu().numpy()

                # Validation & Save
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)

                #⚡ Progress report
                if idx % report_interval == 0:
                    printt(f"now-{len(todo)},all-{idx},{file},{feats.shape}")
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")
