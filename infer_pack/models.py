import math, pdb, os
from time import time as ttime
import torch
from torch import nn
from torch.nn import functional as F
from infer_pack import modules
from infer_pack import attentions
from infer_pack import commons
from infer_pack.commons import init_weights, get_padding
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from infer_pack.commons import init_weights
import numpy as np
from infer_pack import commons


class TextEncoder256(nn.Module):
    """Unit encoder, Embedding-TransformerEncoder-SegFC."""

    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size: int,
        p_dropout,
        f0=True,
    ):
        """
        Args:
            kernel_size - Conformer FF block's conv kernel size (1 means Transformer, not Conformer)
        """
        super().__init__()
        # Common params
        self.hidden_channels = hidden_channels
        # PreNet - Phone/fo Embedding & LReLU
        self.emb_phone = nn.Linear(256, hidden_channels)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        # Main Encoder
        (
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
        ) = (filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        # PostNet - SegFC
        self.out_channels = out_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        """
        Args:
            phone     :: (B, Frame, Feat) - Unit series
            pitch     :: (B, Frame)       - Coarse fo contour
            lengths   :: (B,)             - Effective length of each series in phone/pitch
        Returns:
            mu        :: maybe (B, Feat=o, Frame) - Normal distribution's μ parameter
            log_sigma :: maybe (B, Feat=o, Frame) - Normal distribution's σ parameter
            x_mask    ::       (1,      1, Frame) - Tail-padding mask
        """
        # PreNet - Embedding + scaling
        ## Unit embedding :: (B, Frame, Feat) -> (B, Frame, Feat=h)
        x = self.emb_phone(phone)
        ## Pitch embedding :: (B, Frame) -> (B, Frame, Feat=h)
        if pitch is not None:
            x = x + self.emb_pitch(pitch)
        ## Scaling :: (B, Frame, Feat=h) -> (B, Frame, Feat=h) -> (B, Feat=h, Frame)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)

        ## Tail-padding mask :: (B,) -> (B, Frame) -> (B, Feat=1, Frame)
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, x.size(2)), 1).to(
            x.dtype
        )

        # Encoder :: (B, Feat, Frame) -> (B, Feat, Frame)
        x = self.encoder(x * x_mask, x_mask)

        # PostNet :: (B, Feat, Frame) -> (B, Feat=2*o, Frame) -> 2x (B, Feat=o, Frame)
        stats = self.proj(x) * x_mask
        mu, log_sigma = torch.split(stats, self.out_channels, dim=1)

        return mu, log_sigma, x_mask


class TextEncoder256Sim(nn.Module):
    """Embedding-TransformerEncoder-SegFC."""

    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size: int,
        p_dropout,
        f0=True,
    ):
        """
        Args:
            kernel_size - Conformer FF block's conv kernel size (1 means Transformer, not Conformer)
        """
        super().__init__()
        # Common params
        self.hidden_channels = hidden_channels
        # Phone Embedding
        self.emb_phone = nn.Linear(256, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        # fo embedding
        if f0 == True:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        # Main Encoder
        (
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
        ) = (filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        # Output Projection - SegFC (`out_channels` is not x2, compared to '256')
        self.out_channels = out_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, phone, pitch, lengths):
        """
        Args:
            phone     :: (B, Frame, Feat) - Unit series
            pitch     :: (B, Frame)       - Coarse fo contour
            lengths
        Returns:
            x         :: maybe (B, Feat=o, Frame) - Encoded series
            x_mask    ::       (1,      1, Frame) - Mask
        """
        # Embedding
        if pitch == None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)

        # Encoder
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        ## Tail-padding mask :: (B,) -> (B, Frame) -> (B, Feat=1, Frame)
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask)

        #  PostNet :: (B, Feat=h, Frame) -> (B, Feat=o, Frame)
        x = self.proj(x) * x_mask
        # No split, compared to '256'

        return x, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.n_flows = n_flows
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())
        # Remnants
        (
            self.channels,
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            self.gin_channels,
        ) = (
            channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels,
        )

    def forward(self, x, x_mask, g=None, reverse: bool = False):
        """
        Args:
            reverse - Whether calculate f or f^-1
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    """q(z|x), approximate posterior distribution over latent z given observed x, SegFC-WaveNet-SegFC-NormDist."""

    def __init__(
        self,
        in_channels: int,  # Feature dimension size of observed variable series
        out_channels: int,  # Feature dimension size of latent   variable series (== n_z)
        hidden_channels: int,  # Feature dimension size of hidden layer
        kernel_size: int,  # WaveNet kernel size
        dilation_rate: int,  # WaveNet dilation factor
        n_layers: int,  # WaveNet the number of layers
        gin_channels: int = 0,  # Feature dimension size of global conditioning vector
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # Remnants
        (
            self.in_channels,
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            self.gin_channels,
        ) = (
            in_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels,
        )

    def forward(self, x, x_lengths, g=None):
        """
        Args:
            x         :: (B, Feat=i, Frame) - Observed variable series to be encoded
            x_lengths :: (B,)               - Effective length of each series in `x`
            g         :: (B, Feat,   T=1)   - Time-invariant global conditioning vector
        Returns:
            z         :: maybe (B, Feat=o, Frame) - Latent varible series
            mu        :: maybe (B, Feat=o, Frame) - μ      of conditional normal distribution q(z|x) = N(μ,σ|x)
            log_sigma :: maybe (B, Feat=o, Frame) - log(σ) of conditional normal distribution q(z|x) = N(μ,σ|x)
            x_mask    ::       (B, Feat=1, Frame) - Tail-padding mask
        """
        ## Tail-padding mask :: (B,) -> (B, Frame) -> (B, Feat=1, Frame)
        x_mask = commons.sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        # Transform :: (B, Feat=i, Frame) -> (B, Feat=h, Frame) -> ? -> (B, Feat=2*o, Frame?)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask

        # Normal distribution :: (B, Feat=2*o, Frame?) -> 2x (B, Feat=o, Frame?) -> (B, Feat=o, Frame?) - Reparametrization trick
        mu, log_sigma = torch.split(stats, self.out_channels, dim=1)
        z = (mu + torch.randn_like(mu) * torch.exp(log_sigma)) * x_mask
        return z, mu, log_sigma, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class Generator(torch.nn.Module):
    """Decoder"""

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SineGen(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def forward(self, f0, upp):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (
                    idx + 2
                )  # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
            rad_values = (f0_buf / self.sampling_rate) % 1  ###%1意味着n_har的乘积无法后处理优化
            rand_ini = torch.rand(
                f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device
            )
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
            tmp_over_one = torch.cumsum(rad_values, 1)  # % 1  #####%1意味着后面的cumsum无法再优化
            tmp_over_one *= upp
            tmp_over_one = F.interpolate(
                tmp_over_one.transpose(2, 1),
                scale_factor=upp,
                mode="linear",
                align_corners=True,
            ).transpose(2, 1)
            rad_values = F.interpolate(
                rad_values.transpose(2, 1), scale_factor=upp, mode="nearest"
            ).transpose(
                2, 1
            )  #######
            tmp_over_one %= 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
            sine_waves = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
            )
            sine_waves = sine_waves * self.sine_amp
            uv = self._f02uv(f0)
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=upp, mode="nearest"
            ).transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
        is_half=True,
    ):
        super().__init__()
        self.sine_amp, self.noise_std, self.is_half = sine_amp, add_noise_std, is_half
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )
        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp=None):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        if self.is_half:
            sine_wavs = sine_wavs.half()
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None  # noise, uv


class GeneratorNSF(torch.nn.Module):
    def __init__(
        self,
        initial_channel: int,  # Feature dimension size of input feature series
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,  # Feature dimension size of main block's input series
        upsample_kernel_sizes,
        gin_channels,  # Feature dimension size of speaker vector
        sr,
        is_half=False,
    ):
        super().__init__()
        assert 0 < gin_channels, "gin_channels is now required and should be >0."
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = np.prod(upsample_rates)

        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr, harmonic_num=0, is_half=is_half
        )
        self.noise_convs = nn.ModuleList()

        # PreNet - Feature series convolution & Speaker vector segFC
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, padding="same"
        )
        self.cond = Conv1d(gin_channels, upsample_initial_channel, 1, padding="same")

        # MainNet
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            #                                                                    half channel                                         ,
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))
        self.ups.apply(init_weights)

        # PostNet - Sample series convolution
        self.conv_post = Conv1d(ch, 1, 7, padding="same", bias=False)

    def forward(self, x, f0, g):
        """
        Args:
            x - Feature series
            f0
            g - Time-invariant (global) speaker feature
        """
        assert g is not None, "global speaker conditioning is now required."

        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)

        # PreNet - 'Conv(feat)' + 'SegFC(spk)'
        x = self.conv_pre(x) + self.cond(g)

        for i in range(self.num_upsamples):
            # LReLU-ConvT-.
            #      x------+-MRF-
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            x = x + self.noise_convs[i](har_source)
            xs = None
            # MRF
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # -LReLU-Conv-Tanh
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class SynthesizerTrnMs256NSFsid(nn.Module):
    """Main model w/ fo"""

    def __init__(
        self,
        spec_channels: int,  # Frequency dimension size of spectrogram
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size: int,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim: int,  # The number of speakers in embedding
        gin_channels: int,  # Channel dimension size of speaker embedding vector
        sr,
        **kwargs
    ):
        """
        Args:
            kernel_size - TextEncoder's Conformer FF block conv kernel size (k=1 means SegFC == pure Transformer, not Conformer)
        """
        super().__init__()
        if type(sr) == type("strr"):
            sr = sr2sr[sr]
        # Training params
        self.segment_size = segment_size
        # Global speaker embedding
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        # PriorEncoder - PhoneEncoder/Flow
        self.enc_p = TextEncoder256(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        # PosteriorEncoder
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        # Decoder
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            is_half=kwargs["is_half"],
        )
        # Remnants
        self.inter_channels, self.hidden_channels, self.gin_channels = (
            inter_channels,
            hidden_channels,
            gin_channels,
        )
        (
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
        ) = (filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        (
            self.resblock,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
        ) = (
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
        )
        self.spec_channels = spec_channels
        self.spk_embed_dim = spk_embed_dim

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, phone, phone_lengths, pitch, pitchf, y, y_lengths, ds):
        """
        Args:
            phone         :: (B, Frame, Feat)  - Unit series
            phone_lengths :: (B,)              - Effective length of each series in phone
            pitch         :: (B, Frame)        - Coarse fo contour
            pitchf        :: (B, Frame)        - Fine   fo contour
            y             :: (B, Feat,  Frame) - Observed variable series (e.g. Linear spectrogram)
            y_lengths     :: (B,)              - Effective length of each series in y
            ds            :: (B,)              - Speaker index
        """
        """
        ds -------> [emb_g] -> g -------.-------------------.-----.
        phone --.                       |                   |     |
        pitch --'-> [enc_p] -> μ_p/σ_p  |   z_p <- [flow] <-|     |
                                     y -'-> [enc_q] -.-> z -'-----+-> [dec] -> o
                                                     '-> μ_q/σ_q  |
        pitchf ---------------------------------------------------'
        """
        # Speaker embedding :: (B,) -> (B, Feat) -> (B, Feat, T=1)
        g = self.emb_g(ds).unsqueeze(-1)
        # Unit encoder :: (B, Frame, Feat) & (B, Frame) -> (B, Feat, Frame)
        mu_p, log_sigma_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        # Posterior encoder :: (B, Feat, Frame) -> (B, Feat, Frame)
        z, mu_q, log_sigma_q, y_mask = self.enc_q(y, y_lengths, g)
        z_p = self.flow(z, y_mask, g=g)
        # Decoder :: (B, Feat, Frame=seg) & () & (B, Feat, T=1) -> ()
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        o = self.dec(z_slice, pitchf, g=g)

        return (
            o,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, mu_p, log_sigma_p, mu_q, log_sigma_q),
        )

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        """
        sid -------> [emb_g] -> g ------------------.-------------------.
        phone --.                                   |                   |
        pitch --'--> [enc_p] -> μ/σ -[Norm] -> z_p -'-> [flow^-1] -> z -+-> [dec] -> o
        nsff0 ----------------------------------------------------------'
        """
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], nsff0, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


class SynthesizerTrnMs256NSFsid_nono(nn.Module):
    """Main model w/o fo"""

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim: int,  # The number of speakers in embedding
        gin_channels,  # Feature dimension size of global speaker embedding
        sr=None,
        **kwargs
    ):
        super().__init__()
        # Training params
        self.segment_size = segment_size
        # Common params
        self.inter_channels, self.hidden_channels, self.gin_channels = (
            inter_channels,
            hidden_channels,
            gin_channels,
        )
        # Phone Encoder
        (
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
        ) = (filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.enc_p = TextEncoder256(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=False,
        )
        # Decoder
        (
            self.resblock,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
        ) = (
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        # Posterior Encoder
        self.spec_channels = spec_channels
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        # Flow
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        # Global speaker embedding
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        # Remnants
        self.spk_embed_dim = spk_embed_dim

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, phone, phone_lengths, y, y_lengths, ds):
        """
        Args:
            phone
            phone_lengths
            y
            y_lengths
            ds :: (B, 1)
        """
        """
        ds ----> [emb_g] -> g -------.-------------------.--.
        phone -> [enc_p] -> μ_p/σ_p  |   z_p <- [flow] <-|  |
                                  y -'-> [enc_q] -.-> z -'--'-> [dec] -> o
                                                  '-> μ_q/σ_q
        """
        # Speaker embedding :: (B, 1) -> (B, Feat=256, T=1)
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, phone, phone_lengths, sid, max_len=None):
        """
        sid -------> [emb_g] -> g ------------------.-------------------.
        phone -----> [enc_p] -> μ/σ -[Norm] -> z_p -'-> [flow^-1] -> z -'-> [dec] -> o
        """
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


class SynthesizerTrnMs256NSFsid_sim(nn.Module):
    """No posterior encoder (Not used now)."""

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim: int,  # The number of speakers in embedding
        # hop_length,
        gin_channels: int = 0,  # Feature dimension size of global speaker embedding
        use_sdp=True,
        **kwargs
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.enc_p = TextEncoder256Sim(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            is_half=kwargs["is_half"],
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        # Remnants
        self.spk_embed_dim = spk_embed_dim

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, phone, phone_lengths, pitch, pitchf, y_lengths, ds):
        """
        Args:
            phone
            phone_lengths
            pitch         :: (B, T)
            pitchf
            y_lengths
            ds            :: (B, 1)
        """
        """
        ds --------> [emb_g] -> g -.-------------------.
        phone --.                  |                   |
        pitch --'--> [enc_p] -> x -'-> [flow^-1] -> x -+-> [dec] -> o
        pitchf ----------------------------------------'
        """
        # Speaker embedding :: (B, 1) -> (B, Feat=256, T=1)
        g = self.emb_g(ds).unsqueeze(-1)
        x, x_mask = self.enc_p(phone, pitch, phone_lengths)
        x = self.flow(x, x_mask, g=g, reverse=True)
        z_slice, ids_slice = commons.rand_slice_segments(
            x, y_lengths, self.segment_size
        )
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice

    def infer(self, phone, phone_lengths, pitch, pitchf, ds, max_len=None):
        """
        ds --------> [emb_g] -> g -.-------------------.
        phone --.                  |                   |
        pitch --'--> [enc_p] -> x -'-> [flow^-1] -> x -+-> [dec] -> o
        pitchf ----------------------------------------'
        """
        g = self.emb_g(ds).unsqueeze(-1)  # [b, 256, 1]##1是t，广播的
        x, x_mask = self.enc_p(phone, pitch, phone_lengths)
        x = self.flow(x, x_mask, g=g, reverse=True)
        o = self.dec((x * x_mask)[:, :, :max_len], pitchf, g=g)
        return o, o


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11, 17]
        # periods = [3, 5, 7, 11, 17, 23, 37]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []  #
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            # for j in range(len(fmap_r)):
            #     print(i,j,y.shape,y_hat.shape,fmap_r[j].shape,fmap_g[j].shape)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
