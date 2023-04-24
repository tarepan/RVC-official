import os, traceback
import numpy as np
import torch
import torch.utils.data

from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text


#### DatasetA ##########################################################################################
class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    """Dataset.
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        # Default: 2**15
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.hop_length    = hparams.hop_length     # hop length [sample], must match fo's hop (current configs set properly based on sr)
        # wave2spec parameters
        self.filter_length, self.win_length = hparams.filter_length, hparams.win_length
        # Filtering - audiopaths_and_text == [audiopath, text, pitch, pitchf, dv][]
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.min_text_len = getattr(hparams, "min_text_len",    1) # Minumum frame length (seems to be default value)
        self.max_text_len = getattr(hparams, "max_text_len", 5000) # Maximum frame length (seems to be default value)
        self._filter()

    def _filter(self):
        """Filter text & store spec lengths"""
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, unit, pitch, pitchf, dv in self.audiopaths_and_text:
            # Use data which satisfy `min_text_len <= len(unit) <= max_text_len`
            if self.min_text_len <= len(unit) and len(unit) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, unit, pitch, pitchf, dv])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        """ :: int -> Tensor[1,]"""
        sid = torch.LongTensor([int(sid)])
        return sid

    def get_audio_text_pair(self, audiopath_and_text):
        """Yield datum.

        Args:
            audiopath_and_text - [audiopath, text, pitch, pitchf, dv]
        Returns:
            spec   :: (Freq,      Frame=frm) - Linear-frequency Linear-amplitude spectrogram
            wav    :: (Feat=1,    T=frm*hop) - Waveform, range in [-1, 1]
            phone  :: (Frame=frm, Feat)      - Unit series
            pitch  :: (Frame=frm,)           - Coarse fo series
            pitchf :: (Frame=frm,)           - Fine   fo series
            dv     :: (1,)                   - Speaker index
        """
        # Unpack
        file, phone, pitch, pitchf, dv = audiopath_and_text

        # Load/Generation
        # phone :: (Frame=frm, Feat) / pitch :: (Frame=frm) / pitchf :: (Frame=frm)
        phone, pitch, pitchf = self.get_labels(phone, pitch, pitchf)
        # spec :: (Freq, Frame=frm) / wav :: (1, T=t)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        # Clipping
        len_phone = phone.size()[0]
        len_spec  =  spec.size()[-1]
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            # amor
            spec   = spec[:, :len_min]
            wav    = wav[ :, :len_min * self.hop_length]
            phone  = phone[  :len_min]
            pitch  = pitch[  :len_min]
            pitchf = pitchf[ :len_min]

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(self, phone: str, pitch: str, pitchf: str):
        """
        Args:
            phone  - Path to unit series,       hop_size = 320 (20 msec)
            pitch  - Path to coarse fo contour, hop_size = 160 (10 msec)
            pitchf - Path to  fine  fo contour, hop_size = 160 (10 msec)
        Returns:
            phone  :: (Frame=frm, Feat)  - Unit series, upsampled
            pitch  :: (Frame=frm,)       - Coarse fo series
            pitchf :: (Frame=frm,)       - Fine   fo series
        """
        # Load
        phone, pitch, pitchf = np.load(phone), np.load(pitch), np.load(pitchf)
        # Scale matching - phone 50Hz to 100Hz
        phone = np.repeat(phone, 2, axis=0)
        # Clipping - Variable length from head, maximum = 900 (9 sec), minimum = phone series's length (if <900)
        n_num = min(phone.shape[0], 900)  # DistributedBucketSampler
        phone, pitch, pitchf = phone[:n_num], pitch[:n_num], pitchf[:n_num]
        # Cast
        phone, pitch, pitchf = torch.FloatTensor(phone), torch.LongTensor(pitch), torch.FloatTensor(pitchf)

        return phone, pitch, pitchf

    def get_audio(self, filename: str):
        """Get audio and spectrogram.

        Returns:
            spec       :: (Freq,   Frame) - Linear-frequency Linear-amplitude spectrogram
            audio_norm :: (Feat=1, T)     - Waveform, in range [-1, 1]
        """
        # Load :: (T,)
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")

        # Scaling :: (T,) -> (1, T) - to [-1, 1] NOTE: changed
        audio_norm = audio
        # audio_norm = audio / self.max_wav_value
        # audio_norm = audio / np.abs(audio).max()
        audio_norm = audio_norm.unsqueeze(0)

        # Spec
        spec_filename = filename.replace(".wav", ".spec.pt")
        spec_file_exist = os.path.exists(spec_filename)
        spec_loaded = False
        if spec_file_exist:
            try:
                spec = torch.load(spec_filename)
                spec_loaded = True
            except:
                # Load failed
                spec_loaded = False
                print(spec_filename, traceback.format_exc())
        if (not spec_file_exist) or (not spec_loaded):
            # spec :: (B=1, T) -> (B=1, Freq, Frame) -> (Freq, Frame) - Linear-frequency Linear-amplitude spectrogram
            spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False).squeeze(0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)

        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollateMultiNSFsid:
    """Collate function, Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch with tail-padding.

        Args:
            batch :: [items]
                #0 spec   :: (Feat,   Frame) - Linear-frequency Linear-amplitude spectrogram
                #1 wave   :: (Feat=1, T)     - Waveform
                #2 phone  :: (Frame,  Feat)  - Unit series
                #3 pitch  :: (Frame,)        - Coarse fo contour
                #4 pitchf :: (Frame,)        - Fine   fo contour
                #5 sid    :: (1,)            - Speaker index
        Returns:
            phone_padded  :: (B, Frame,  Feat)  - Tail-padded unit series
            phone_lengths :: (B,)               - Effective length of each series in phone_padded
            pitch_padded  :: (B, Frame)         - Tail-padded pitch  contour
            pitchf_padded :: (B, Frame)         - Tail-padded pitchf contour
            spec_padded   :: (B, Feat,   Frame) - Tail-padded Linear-frequency Linear-amplitude spectrogram
            spec_lengths  :: (B,)               - Effective length of each series in spec_padded
            wave_padded   :: (B, Feat=1, T)     - Tail-padded waveforms
            wave_lengths  :: (B,)               - Effective length of each series in wave_padded
            sid           :: (B,)               - Speaker index
        """
        # Lengh-sorted Indice - e.g. batch -> [20, 40, 30] -> [1, 2, 0]
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True)

        # Padded placeholder - full size matrix with zeros
        ## spec_padded :: (B, Feat, Frame=frame_max)
        max_spec_len = max([x[0].size(1) for x in batch])
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        spec_padded.zero_()
        ## spec_lengths :: (B,)
        spec_lengths = torch.LongTensor(len(batch))
        ## wave_padded :: (B, Feat=1, T=t_max)
        max_wave_len = max([x[1].size(1) for x in batch])
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        wave_padded.zero_()
        ## wave_lengths :: (B,)
        wave_lengths = torch.LongTensor(len(batch))
        ## phone_padded :: (B, Frame=frame_max, Feat)
        max_phone_len = max([x[2].size(0) for x in batch])
        phone_padded = torch.FloatTensor(len(batch), max_phone_len, batch[0][2].shape[1])
        phone_padded.zero_()
        ## phone_lengths :: (B,)
        phone_lengths = torch.LongTensor(len(batch))
        ## pitch_padded :: (B, Frame=frame_max)
        pitch_padded = torch.LongTensor(len(batch), max_phone_len)
        pitch_padded.zero_()
        ## pitchf_padded :: (B, Frame=frame_max)
        pitchf_padded = torch.FloatTensor(len(batch), max_phone_len)
        pitchf_padded.zero_()
        ## sid :: (B,)
        sid = torch.LongTensor(len(batch))

        # Insert datums into fullsize zero matrix (results in tail zeros)
        for i in range(len(ids_sorted_decreasing)):
            # from longest datum
            row = batch[ids_sorted_decreasing[i]]

            # spec_padded :: (B, Feat, Frame=frame_max) <- (Feat, Frame)
            spec = row[0]
            len_spec = spec.size(1)
            spec_padded[i, :, :len_spec] = spec
            spec_lengths[i] = len_spec
            # wave_padded :: (B, Feat=1, T=t_max) <- (Feat=1, T)
            wave = row[1]
            len_wave = wave.size(1)
            wave_padded[i, :, :len_wave] = wave
            wave_lengths[i] = len_wave
            ## phone_padded :: (B, Frame=frame_max, Feat) <- (Frame, Feat)
            phone = row[2]
            len_phone = phone.size(0)
            phone_padded[i, :len_phone, :] = phone
            phone_lengths[i] = len_phone
            ## pitch_padded :: (B, Frame=frame_max) <- (Frame,)
            pitch = row[3]
            pitch_padded[ i, :pitch.size(0)] = pitch
            ## pitchf_padded :: (B, Frame=frame_max) <- (Frame,)
            pitchf = row[4]
            pitchf_padded[i, :pitchf.size(0)] = pitchf
            ## sid :: (B,) <- (1,)
            sid[i] = row[5]

        return (
            phone_padded, phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,  spec_lengths,
            wave_padded,  wave_lengths,
            sid,
        )
#### /DatasetA #########################################################################################


#### DatasetB ##########################################################################################
class TextAudioLoader(torch.utils.data.Dataset):
    """Dataset.
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text, dv in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text, dv])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        dv = audiopath_and_text[2]

        phone = self.get_labels(phone)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min, :]
        return (spec, wav, phone, dv)

    def get_labels(self, phone):
        phone = np.load(phone)
        phone = np.repeat(phone, 2, axis=0)
        n_num = min(phone.shape[0], 900)  # DistributedBucketSampler
        phone = phone[:n_num, :]
        phone = torch.FloatTensor(phone)
        return phone

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio
#        audio_norm = audio / self.max_wav_value
#        audio_norm = audio / np.abs(audio).max()

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename)
            except:
                print(spec_filename, traceback.format_exc())
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        spec_padded.zero_()
        wave_padded.zero_()

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(
            len(batch), max_phone_len, batch[0][2].shape[1]
        )
        phone_padded.zero_()
        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            sid[i] = row[3]

        return (
            phone_padded,
            phone_lengths,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )
#### /DatasetB #########################################################################################


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """Common sampler.
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):  #
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
