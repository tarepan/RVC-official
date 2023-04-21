"""
Load pre-resampled 16kHz audios, then convert them to coarse/fine fo series.

[Before Run]
    {exp_dir}/
        extract_f0_feature.log  # Not required, but can exist
        1_16k_wavs/
            {xxx}.wav           # input, pre-resampled 16kHz audio

[After Run]
    {exp_dir}/
        extract_f0_feature.log  # Appended
        1_16k_wavs/
            {xxx}.wav           # No change
        2a_f0/
            {xxx}.wav.npy       # Output, generated Coarse fo series
        2b-f0nsf/
            {xxx}.wav.npy       # Output, generated   Fine fo series
"""

import os, traceback, sys, logging
from multiprocessing import Process

import numpy as np
import librosa
import parselmouth
import pyworld


logging.getLogger("numba").setLevel(logging.WARNING)

# Args
exp_dir: str  =     sys.argv[1]
n_p: int      = int(sys.argv[2])
f0method: str =     sys.argv[3]

# Logger
f = open(f"{exp_dir}/extract_f0_feature.log", "a+")
def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


class FeatureInput(object):
    """Coarse/Fine fo extractor."""
    def __init__(self, samplerate: int = 16000, hop_size: int = 160):
        """
        Args:
            samplerate - Audio target sampling rate (for resampling)
            hop_size
        """
        self.fs = samplerate
        self.hop = hop_size

        # fo min/max configuration
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # The number of coarse fo bins
        self.f0_bin = 256

    def compute_f0(self, path, f0_method: str):
        """
        Args:
            path      - Path to the audio .wav file
            f0_method - "pm" | "harvest" | "dio"
        Returns:
            f0 :: (Frame,)
        """
        # Load waveform
        x, sr = librosa.load(path, self.fs)
        p_len = x.shape[0] // self.hop
        assert sr == self.fs

        # Extraction
        ## Praat
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            # Same as `self.f0_min`/`self.f0_max` but Int
            f0_min, f0_max = 50, 1100
            f0 = (
                parselmouth.Sound(x, sr)
                .to_pitch_ac(time_step=time_step / 1000, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max)
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                # zero-padding
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")

        ## Harvest + Stonemask
        # NOTE: Stonemask is not needed for Harvest, isn't it...?
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(x.astype(np.double), fs=sr, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=1000 * self.hop / sr)
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)

        ## DIO + Stonemask
        elif f0_method == "dio":
            f0, t = pyworld.dio(    x.astype(np.double), fs=sr, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=1000 * self.hop / sr)
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)

        return f0

    def coarse_f0(self, f0):
        """Fine-fo to Coarse-fo.

        Args:
            f0        :: (Frame,) - Fine fo
        Returns:
            f0_coarse :: (Frame,) - Coarse fo
        """
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1

        # Rounding
        ## (0, 1] -> bin#1
        f0_mel[f0_mel <= 1] = 1
        ## (f0_bin-1, +∞) -> bin#{f0_bin-1}
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        ## [1, f0_bin-1] -> int
        f0_coarse = np.rint(f0_mel).astype(np.int)

        # Validation
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min(),)

        return f0_coarse

    def go(self, paths, f0_method: str):
        """
        Args:
            paths
            f0_method              - "pm" | "harvest" | "dio"
        Outputs:
            coarse_pit :: (Frame,) - Coarse fo at `opt_path1` (e.g. f"{exp_dir}/2a_f0/{xxx}.wav.npy")
            featur_pit :: (Frame,) - Fine   fo at `opt_path2` (e.g. f"{exp_dir}/2b-f0nsf/{xxx}.wav.npy")
        """
        if len(paths) == 0:
            printt("no-f0-todo")
        else:
            printt(f"todo-f0-{len(paths)}")
            log_interval = max(len(paths) // 5, 1)
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    # Already-exist check
                    if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
                        continue

                    # Fine fo :: (Frame,) - For NSF
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(opt_path2, featur_pit, allow_pickle=False)

                    # Coarse fo :: (Frame,) - For ori
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(opt_path1, coarse_pit, allow_pickle=False)

                    # Logging
                    if idx % log_interval == 0:
                        printt(f"f0ing,now-{idx},all-{len(paths)},-{inp_path}")
                except:
                    printt(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}")


if __name__ == "__main__":
    printt(sys.argv)

    # I/O directories
    inp_root  = f"{exp_dir}/1_16k_wavs" # .wav
    opt_root1 = f"{exp_dir}/2a_f0"      # Coarse fo
    opt_root2 = f"{exp_dir}/2b-f0nsf"   # Fine   fo
    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    # List up paths directly under `inp_root`
    ## paths :: (input_path, output_path1, output_path2)[]
    paths = []
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = f"{inp_root}/{name}"
        if "spec" in inp_path:
            continue
        # e.g. f"{exp_dir}/2a_f0/{xxx}.wav"
        opt_path1 = f"{opt_root1}/{name}"
        opt_path2 = f"{opt_root2}/{name}"
        paths.append([inp_path, opt_path1, opt_path2])

    # Run fo extraction w/ multi-thread
    featureInput = FeatureInput()
    ps = []
    for i in range(n_p):
        p = Process(target=featureInput.go, args=(paths[i::n_p], f0method,))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
