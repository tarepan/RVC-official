"""
Load raw audios, then modify and resample.

Processes:
  1. Resampling fs=`sr`
  2. HPF 48Hz
  3. Silence clip
  3. Chunk
  4. Volume normalize
  5. Preemphasis?
  6. s16-nize (-> output1)
  7. Resampling fs=16k (-> output2)

[Before Run]
    {inp_root}/
        {xxx}.wav                       # Input waveform
    {exp_dir}/
        preprocess.log                  # Not required, but can exist

[After Run]
    {inp_root}/
        {xxx}.wav                       # No change
    {exp_dir}/
        preprocess.log                  # Appended
        0_gt_wavs/
            {file_idx}_{slice_idx}.wav  # Output, processed and resampled waveform, srHz/s16
        1_16k_wavs/
            {file_idx}_{slice_idx}.wav  # Output, processed and resampled waveform, 16kHz/s16
"""

import sys, os, traceback, multiprocessing

import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa

from slicer2 import Slicer
from my_utils import load_audio


# Configs
now_dir = os.getcwd()
sys.path.append(now_dir)
## Args
inp_root: str = sys.argv[
    1
]  # Data root, directly under which raw .wav files should exist
sr: int = int(sys.argv[2])  # Audio target sampling rate, used for resampling
n_p: int = int(sys.argv[3])  # The number of process for MP
exp_dir: str = sys.argv[
    4
]  # Experiment directory, under which preprocessed waveforms are saved
noparallel: bool = sys.argv[5] == "True"  # Whether to preprocess non-parallelly

# Logger
mutex = multiprocessing.Lock()
f = open(f"{exp_dir}/preprocess.log", "a+")


def println(strr):
    mutex.acquire()
    print(strr)
    f.write("%s\n" % strr)
    f.flush()
    mutex.release()


class PreProcess:
    def __init__(self, sr: int, exp_dir: str):
        """
        Args:
            sr - Target sampling rate of '0_gt_wavs'
        """
        self.slicer = Slicer(
            sr=sr,
            threshold=-40,
            min_length=800,
            min_interval=400,
            hop_size=15,
            max_sil_kept=150,
        )
        self.sr = sr

        # Filter - Parameters of 48Hz high-pass filter
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=sr)

        self.per = 3.7  # Standard chunk length  [sec]
        self.overlap = 0.3  # Chunk-to-Chunk overlap [sec]
        self.tail = self.per + self.overlap
        self.max = 0.95
        self.alpha = 0.8
        self.gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
        self.wavs16k_dir = f"{exp_dir}/1_16k_wavs"
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)
        # Remnant
        self.exp_dir = exp_dir

    def norm_write(self, tmp_audio, file_idx: int, slice_idx: int):
        """Write out normalized audio (preemphasized & resampled).
        Args:
            tmp_audio :: (T,) - Audio waveform, in range [-1, 1] with sr=self.sr
            file_idx          - File index in a input directory
            slice_idx         - Slice index in a waveform
        Outputs:
            "{gt_wavs_dir}/{file_idx}_{slice_idx}.wav" - XX             waveform slice, s16/srHz
            "{wavs16k_dir}/{file_idx}_{slice_idx}.wav" - XX & resampled waveform slice, s16/16kHz
        """
        filename = f"{file_idx}_{slice_idx}.wav"

        # NOTE: Amplitude normalize & preemphasis...?
        # NOTE: Audible difference seems to be very small (loudness is ofcourse changed)
        tmp_audio = (tmp_audio / np.abs(tmp_audio).max() * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            f"{self.gt_wavs_dir}/{filename}", self.sr, tmp_audio.astype(np.float32)
        )

        # Saved as sint16/16kHz
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )  # , res_type="soxr_vhq"
        wavfile.write(
            f"{self.wavs16k_dir}/{filename}", 16000, tmp_audio.astype(np.float32)
        )

    def pipeline(self, path_ipt: str, file_idx: int):
        """
        Args:
            path_ipt - Path to input .wav file candidate
            file_idx - File index in a data root
        """
        try:
            # 1. audio :: (T,) - in range [-1, 1], sr=self.sr
            audio = load_audio(path_ipt, self.sr)

            # 2. Filtering - Cut below 48Hz
            ## zero phased digital filter cause pre-ringing noise...
            audio = signal.lfilter(self.bh, self.ah, audio)

            # 3. Silence clipping :: (T,) -> List[NDArray[(L,)]] - Convert an audio into continuous slices (no silence within a slice)
            slices = self.slicer.slice(audio)

            # 4. Chunking & 5-7. Processing
            total_chunk_idx = 0  # Chunk index in a audio
            for audio_slice in slices:
                slice_chunk_idx = 0  # Chunk index in a slice
                while True:
                    chunk_hop_sec = self.per - self.overlap  # [sec]
                    start = int(self.sr * chunk_hop_sec * slice_chunk_idx)
                    len_chunk = int(self.per * self.sr)
                    end = start + len_chunk
                    slice_chunk_idx += 1
                    if len(audio_slice[start:]) > self.tail * self.sr:
                        chunk = audio_slice[start:end]
                        self.norm_write(chunk, file_idx, total_chunk_idx)
                        total_chunk_idx += 1
                    else:
                        # Tail
                        chunk = audio_slice[start:]
                        total_chunk_idx += 1
                        break
                # Preprocess - Tail
                self.norm_write(chunk, file_idx, total_chunk_idx)
            println(f"{path_ipt}->Suc.")

        except:
            # `path_ipt` specifying Non-wav file or directory come here
            println(f"{path_ipt}->{traceback.format_exc()}")

    def pipeline_mp(self, infos):
        """⚡ Preprocess several data on a process.
        Args:
            infos
                path_ipt :: str - Path to the input file candidate (no guarantee of .wav or even file (could be dir))
                file_idx :: int - File index
        """
        for path_ipt, file_idx in infos:
            self.pipeline(path_ipt, file_idx)

    def pipeline_mp_inp_dir(self, inp_root: str, n_p: int):
        """⚡ Run multi-process preprocessing"""
        try:
            # infos :: (path_ipt :: str, file_idx :: int)[] - List up all items directly under the `inp_root`
            infos = [
                (f"{inp_root}/{file_name}", file_idx)
                for file_idx, file_name in enumerate(sorted(list(os.listdir(inp_root))))
            ]

            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    p.start()
                    ps.append(p)
                    for p in ps:
                        p.join()
        except:
            println(f"Fail. {traceback.format_exc()}")


def preprocess_trainset(inp_root: str, sr: int, n_p: int, exp_dir: str):
    """⚡ Execute runner."""
    pp = PreProcess(sr, exp_dir)
    println("start preprocess")
    println(sys.argv)
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("end preprocess")


if __name__ == "__main__":
    # ⚡
    preprocess_trainset(inp_root, sr, n_p, exp_dir)
