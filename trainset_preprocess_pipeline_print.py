import sys,os,multiprocessing
now_dir=os.getcwd()
sys.path.append(now_dir)

# Args
inp_root = sys.argv[1]             # (maybe) data root?
sr = int(sys.argv[2])              # Audio sampling rate
n_p = int(sys.argv[3])             # (maybe) The number of process for MP
exp_dir = sys.argv[4]              # Experiment directory, under which preprocessed waveforms are saved
noparallel = sys.argv[5] == "True" # Whether to preprocess non-parallelly

import numpy as np,os,traceback
from slicer2 import Slicer
import librosa,traceback
from  scipy.io import wavfile
import multiprocessing
from my_utils import load_audio

mutex = multiprocessing.Lock()
f = open("%s/preprocess.log"%exp_dir, "a+")
def println(strr):
    mutex.acquire()
    print(strr)
    f.write("%s\n" % strr)
    f.flush()
    mutex.release()


class PreProcess():
    def __init__(self, sr, exp_dir):
        self.slicer = Slicer(sr=sr, threshold=-32, min_length=800, min_interval=400, hop_size=15, max_sil_kept=150)
        self.sr = sr
        self.per = 3.7
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.95
        self.alpha = 0.8
        # {exp_dir}/
        #     0_gt_wavs/
        #         {idx0}_{idx1}.wav
        #     1_16k_wavs/
        #         {idx0}_{idx1}.wav
        self.gt_wavs_dir = "%s/0_gt_wavs"%exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs"%exp_dir
        os.makedirs(exp_dir,          exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)
        # Remnant
        self.exp_dir = exp_dir

    def norm_write(self, tmp_audio, idx0: int, idx1: int):
        """Write out normalized audio (preemphasized & resampled).
        Args:
            tmp_audio - Audio waveform
            idx0      - used for .wav name
            idx1      - used for .wav name
        Outputs:
            "{gt_wavs_dir}/{idx0}_{idx1}.wav" - XX             waveform, s16/16kHz
            "{wavs16k_dir}/{idx0}_{idx1}.wav" - XX & resampled waveform, s16/16kHz

        """
        # NOTE: preemphasis ...?
        tmp_audio = (tmp_audio / np.abs(tmp_audio).max() * (self.max * self.alpha)) + (1 - self.alpha) * tmp_audio

        wavfile.write("%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1), self.sr, (tmp_audio*32768).astype(np.int16))

        # Saved as sint16/16kHz
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write("%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1), 16000,   (tmp_audio*32768).astype(np.int16))

    def pipeline(self, path: str, idx0: int):
        """Run preprocessing."""
        try:
            audio = load_audio(path, self.sr)
            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while (1):
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if (len(audio[start:]) > self.tail * self.sr):
                        tmp_audio = audio[start:start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            println("%s->Suc."%path)
        except:
            println("%s->%s"%(path, traceback.format_exc()))

    def pipeline_mp(self,infos):
        """Preprocess several data on a process."""
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            # (f"{inp_root}/{name}", idx) for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            infos = [("%s/%s" % (inp_root, name), idx) for idx, name in enumerate(sorted(list(os.listdir(inp_root))))]
            # global flag
            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps=[]
                for i in range(n_p):
                    p = multiprocessing.Process(target=self.pipeline_mp, args=(infos[i::n_p],))
                    p.start()
                    ps.append(p)
                    for p in ps:
                        p.join()
        except:
            println("Fail. %s"%traceback.format_exc())


def preprocess_trainset(inp_root, sr, n_p, exp_dir):
    pp = PreProcess(sr, exp_dir)

    println("start preprocess")
    println(sys.argv)
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("end preprocess")


if __name__=='__main__':
    preprocess_trainset(inp_root, sr, n_p, exp_dir)
