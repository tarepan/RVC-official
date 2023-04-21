"""
Load raw audios, then modify and resample.

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
            {file_idx}_{slice_idx}.wav  # Output, preprocessed waveform, srHz/s16
        1_16k_wavs/
            {file_idx}_{slice_idx}.wav  # Output, preprocessed waveform, 16kHz/s16
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
inp_root:   str  =     sys.argv[1]           # Data root, directly under which raw .wav files should exist
sr:         int  = int(sys.argv[2])          # Audio sampling rate
n_p:        int  = int(sys.argv[3])          # (maybe) The number of process for MP
exp_dir:    str  =     sys.argv[4]           # Experiment directory, under which preprocessed waveforms are saved
noparallel: bool =     sys.argv[5] == "True" # Whether to preprocess non-parallelly

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
        self.slicer = Slicer(sr=sr, threshold=-40, min_length=800, min_interval=400, hop_size=15, max_sil_kept=150)
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = 3.7
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.95
        self.alpha = 0.8
        self.gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
        self.wavs16k_dir = f"{exp_dir}/1_16k_wavs"
        os.makedirs(exp_dir,          exist_ok=True)
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
        tmp_audio = (tmp_audio / np.abs(tmp_audio).max() * (self.max * self.alpha)) + (1 - self.alpha) * tmp_audio
        wavfile.write(f"{self.gt_wavs_dir}/{filename}", self.sr, (tmp_audio * 32768).astype(np.int16))

        # Saved as sint16/16kHz
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write(f"{self.wavs16k_dir}/{filename}", 16000,   (tmp_audio * 32768).astype(np.int16))

    def pipeline(self, path_ipt: str, file_idx: int):
        """
        Args:
            path_ipt - Path to input .wav file candidate
            file_idx - File index in a data root
        """
        try:
            # audio :: (T,) - in range [-1, 1], sr=self.sr
            audio = load_audio(path_ipt, self.sr)

            audio = signal.filtfilt(self.bh, self.ah, audio)

            # Slice & Preprocess
            slice_idx = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        # Slice
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        # Preprocess
                        self.norm_write(tmp_audio, file_idx, slice_idx)
                        slice_idx += 1
                    else:
                        # Slice - tail
                        tmp_audio = audio[start:]
                        break
                # Preprocess - Tail
                self.norm_write(tmp_audio, file_idx, slice_idx)
            println(f"{path_ipt}->Suc.")

        except:
            # `path_ipt` specifying Non-wav file or directory come here
            println(f"{path_ipt}->{traceback.format_exc()}")

    def pipeline_mp(self, infos):
        """Preprocess several data on a process.
        Args:
            infos
                path_ipt :: str - Path to the input file candidate (no guarantee of .wav or even file (could be dir))
                file_idx :: int - File index
        """
        for path_ipt, file_idx in infos:
            self.pipeline(path_ipt, file_idx)

    def pipeline_mp_inp_dir(self, inp_root: str, n_p: int):
        try:
            # infos :: (path_ipt :: str, file_idx :: int)[] - List up all items directly under the `inp_root`
            infos = [(f"{inp_root}/{file_name}", file_idx) for file_idx, file_name in enumerate(sorted(list(os.listdir(inp_root))))]

            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(target=self.pipeline_mp, args=(infos[i::n_p],))
                    p.start()
                    ps.append(p)
                    for p in ps:
                        p.join()
        except:
            println(f"Fail. {traceback.format_exc()}")


def preprocess_trainset(inp_root: str, sr: int, n_p: int, exp_dir: str):
    pp = PreProcess(sr, exp_dir)
    println("start preprocess")
    println(sys.argv)
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("end preprocess")


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir)
