import ffmpeg
import numpy as np


def load_audio(file, sr):
    """Load audio from the file as amplitude [-1, +1] & sr [Hz]"""
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        # PCM signed 16-bit little-endian, mono (down-mixing), sr-specified (resampling?)
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    # s16/sr -> ±1/sr
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
