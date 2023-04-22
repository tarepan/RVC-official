import numpy as np, parselmouth, torch
from time import time as ttime
import torch.nn.functional as F
from config import x_pad, x_query, x_center, x_max
import scipy.signal as signal
import pyworld, os, traceback, faiss
from scipy import signal

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


class VC(object):
    "Inference runner"
    def __init__(self, tgt_sr, device, is_half):
        self.sr        = 16000               # hubert输入采样率
        self.window    =   160               # 每帧点数 [samples/frame]
        self.t_pad     = self.sr * x_pad     # 每条前后pad时间
        self.t_pad_tgt = tgt_sr  * x_pad
        self.t_pad2    = self.t_pad * 2
        self.t_query   = self.sr * x_query   # 查询切点前后查询时间
        self.t_center  = self.sr * x_center  # 查询切点位置
        self.t_max     = self.sr * x_max     # 免查询时长阈值
        self.device    = device
        self.is_half   = is_half

    def get_f0(self, x, p_len, f0_up_key, f0_method, inp_f0=None):
        """
        Args:
            inp_f0
        Returns:
            f0_coarse - Coarse fo contour
            f0bak     -  Fine  fo contour
        """
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        # Extraction
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(time_step=time_step / 1000, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max)
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(x.astype(np.double), fs=self.sr, f0_ceil=f0_max, f0_floor=f0_min, frame_period=10)
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
            f0 = signal.medfilt(f0, 3)

        # Conversion
        f0 *= pow(2, f0_up_key / 12)

        # 
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[x_pad * tf0 : x_pad * tf0 + len(replace_f0)].shape[0]
            f0[x_pad * tf0 : x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]
        f0bak = f0.copy()

        # Coarse fo contour
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)

        return f0_coarse, f0bak

    def vc(self, model, net_g, sid, audio0, pitch, pitchf, times, index, big_npy, index_rate):
        """Convert voice.

        Args:
            model      - Wave-to-Feature model (e.g. HuBERT)
            net_g      - Feature-to-Wave model
            sid        - Target speaker's ID
            audio0     - Source audio waveform
            pitch      - Converted coarse pitch contour
            pitchf     - Converted  fine  pitch contour
            times      - Output of Time to VC
            index      - (Retrieval related something)
            big_npy    - (Retrieval related something)
            index_rate
        """
        # Load
        feats = torch.from_numpy(audio0)
        feats = feats.half() if self.is_half else feats.float()

        # Stereo-to-Mono
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()

        # Prepare HuBERT inputs - feats :: (B=1, T)
        feats = feats.view(1, -1)
        mask_placeholder = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        inputs = {
            "source": feats.to(self.device),
            "padding_mask": mask_placeholder,
            "output_layer": 9,
        }

        t0 = ttime()

        # Audio-to-Unit :: (B=1, T) -> (B=1, Frame, Feat)
        with torch.no_grad():
            feats = model.final_proj(model.extract_features(**inputs)[0])

        # Retrieval ...?
        if (isinstance(index, type(None)) == False) and (isinstance(big_npy, type(None)) == False) and index_rate != 0:
            npy = feats[0].cpu().numpy()
            npy = npy.astype("float32") if self.is_half else npy
            _, I = index.search(npy, 1)
            npy = big_npy[I.squeeze()]
            npy = npy.astype("float16") if self.is_half else npy
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats

        # Upsampling - 50Hz to 100Hz
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        t1 = ttime()

        # Feature Length matching
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch  = pitch[ :, :p_len]
                pitchf = pitchf[:, :p_len]
        p_len = torch.tensor([p_len], device=self.device).long()

        # Feat2Audio
        with torch.no_grad():
            if pitch != None and pitchf != None:
                audio1 = net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0] * 32768
            else:
                audio1 = net_g.infer(feats, p_len,                sid)[0][0, 0] * 32768
            audio1 = audio1.data.cpu().float().numpy().astype(np.int16)

        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1

        # Clean up
        del feats, p_len, mask_placeholder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        file_big_npy,
        index_rate,
        if_f0,
        f0_file=None,
    ):
        """Runner function."""
        if file_big_npy != "" and file_index != "" and os.path.exists(file_big_npy) and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = np.load(file_big_npy)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None

        # Audio preprocessing
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                # [0 : -win] + [1 : -(win-1)] + ... + [win-1 : -1]
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                # t_center, 2*t_center, ...
                opt_ts.append(
                    t - self.t_query + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )

        # Pitch
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        ## Load pitch file
        inp_f0 = None
        if hasattr(f0_file, "name") == True:
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        ## Pitch extraction and conversion
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key, f0_method, inp_f0)
            pitch, pitchf = pitch[ :p_len], pitchf[:p_len]
            pitch  = torch.tensor(pitch,  device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        # Speaker ID :: (B=1, 1)
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        t2 = ttime()
        times[1] += t2 - t1

        # VC - Chunk-wise conversion
        s, t = 0, None
        audio_opt = []
        for t in opt_ts:
            s_f = s // self.window
            t = t // self.window * self.window
            audio_opt.append(
                self.vc(model, net_g, sid,
                    audio_pad[s : t + self.t_pad2 + self.window],
                    pitch[ :, s_f : (t + self.t_pad2) // self.window] if if_f0 == 1 else None,
                    pitchf[:, s_f : (t + self.t_pad2) // self.window] if if_f0 == 1 else None,
                    times,
                    index, big_npy, index_rate,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
            s = t
        # tail?
        t_f = t // self.window
        audio_opt.append(
            self.vc(
                model, net_g, sid,
                audio_pad[t:],
                (pitch[ :, t_f:] if t is not None else pitch ) if if_f0 == 1 else None,
                (pitchf[:, t_f:] if t is not None else pitchf) if if_f0 == 1 else None,
                times,
                index, big_npy, index_rate,
            )[self.t_pad_tgt : -self.t_pad_tgt]
        )

        # Non-overlap Add
        audio_opt = np.concatenate(audio_opt)

        # Clean up
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_opt
