import os
import glob
import sys
import argparse
import logging
import json
import subprocess

import numpy as np
from scipy.io.wavfile import read
import torch


# ⚡
MATPLOTLIB_FLAG = False
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    """⚡"""
    # Validation
    assert os.path.isfile(checkpoint_path)

    # Load
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    # Restore
    ## Model
    saved_state_dict = checkpoint_dict["model"]
    ### Merge restored state and init state
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # 模型需要的shape
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                print("shape-%s-mismatch|need-%s|get-%s" % (k, state_dict[k].shape, saved_state_dict[k].shape))
                raise KeyError
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v  # 模型自带的随机值
    ### Run restore
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")
    ## Epoch/LR/Optim
    iteration     = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")

    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    """⚡
    Outputs:
        {
            model                                     - Model state dict
            iteration                                 - epoch
            optimizer :: {}
                state     :: {}                           - current optimization state
                    N
                        step                                  - The number of global step
                        exp_avg
                        exp_avg_sq
                param_groups :: [parameter group::{}]     - (show only some attributes)
                    lr           :: float                     - Current learning rate, modified by scheduler
                    betas        :: (float, float)
                    weight_decay :: flaot
                    initial_lr   :: float                     - Initial learning rate
            learning_rate :: float                    - initial learning rate (`hps.train.learning_rate`)
        }
    """
    logger.info("Saving model and optimizer state at epoch {} to {}".format(iteration, checkpoint_path))
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path, # f"{hps.model_dir}/G_{global_step|2333333}.pth"
    )

def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    """⚡"""
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """⚡"""
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    """⚡"""
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    """Load data list."""
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    """
    todo:
      结尾七人组：
        保存频率、总epoch                     done
        bs                                    done
        pretrainG、pretrainD                  done
        卡号：os.en["CUDA_VISIBLE_DEVICES"]   done
        if_latest                             todo
      模型：if_f0                             todo
      采样率：自动选择config                  done
      是否缓存数据集进GPU:if_cache_data_in_gpu done

      -m:
        自动决定training_files路径,改掉train_nsf_load_pretrain.py里的hps.data.training_files    done
      -c不要了
    """
    # Load arguments
    # NOTE: important for base training: `total_epoch` / `batch_size` / `if_f0`
    parser = argparse.ArgumentParser()
    parser.add_argument("-se", "--save_every_epoch",     type=int, required=True,              help="checkpoint save frequency (epoch)")
    parser.add_argument("-te", "--total_epoch",          type=int, required=True,              help="total_epoch")
    parser.add_argument("-pg", "--pretrainG",            type=str,                default="",  help="Pretrained Discriminator path")
    parser.add_argument("-pd", "--pretrainD",            type=str,                default="",  help="Pretrained Generator path")
    parser.add_argument("-g",  "--gpus",                 type=str,                default="0", help="split by -")
    parser.add_argument("-bs", "--batch_size",           type=int, required=True,              help="batch size")
    parser.add_argument("-e",  "--experiment_dir",       type=str, required=True,              help="experiment dir")
    parser.add_argument("-sr", "--sample_rate",          type=str, required=True,              help="sample rate, 32k/40k/48k")
    parser.add_argument("-f0", "--if_f0",                type=int, required=True,              help="use f0 as one of the inputs of the model, 1 or 0")
    parser.add_argument("-l",  "--if_latest",            type=int, required=True,              help="if only save the latest G/D pth file, 1 or 0")
    parser.add_argument("-c",  "--if_cache_data_in_gpu", type=int, required=True,              help="if caching the dataset in GPU memory, 1 or 0")
    args = parser.parse_args()

    # Prepare directories
    experiment_dir = os.path.join("./logs", args.experiment_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Load and Save config file
    """
    Configs
    {                                                                                | What's for                                          | compared to VITS
        "train": {
            "log_interval": 200,                                                                                                           | -
            "seed": 1234,                                                                                                                  | same
            "epochs": 20000,                                                                                                               | x2
            "learning_rate": 1e-4,                                                                                                         | half
            "betas": [0.8, 0.99],                                                                                                          | same
            "eps": 1e-9,                                                                                                                   | same
            "batch_size": 4,                                                                                                               | 64
            "fp16_run": true,                                                                                                              | same
            "lr_decay": 0.999875,                                                                                                          | same
            "segment_size": 12800,                                                                                                         |
            "init_lr_ratio": 1,                                                      | Not used for fine-tuning                            | same 
            "warmup_epochs": 0,                                                      | Not used for fine-tuning                            | same
            "c_mel": 45,                                                             | loss balance                                        | same
            "c_kl": 1.0,                                                             | loss balance                                        | same
        },
        "data": {
            "max_wav_value":  32768.0, -                                             | Not used now                                        | same
            "sampling_rate":  32000,   - Ground-truth waveform's sampling rate       | many purposes                                       | <-22050
            "filter_length":   1024,   - STFT n_fft                                  | mel-loss & linear-spec ipt & model input shape
            "hop_length":       320,   - Feature hop size [sample/frame]             | mel-loss & linear-spec ipt & others (clipping, segment_size, etc...)
            "win_length":      1024,   - STFT window length                          | mel-loss & linear-spec ipt
            "n_mel_channels":    80,   - Frequency dimension size of mel-spectrogram | mel-loss                                            | same
            "mel_fmin":           0.0, - Lowest  frequency in mel-spectrogarm        | mel-loss                                            | same
            "mel_fmax":        null,   - Highest frequency in mel-spectrogarm        | mel-loss                                            | same
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0,                                           | <- 0.1
            "resblock": "1",
            "resblock_kernel_sizes": [3,7,11],
            "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
            "upsample_rates": [10,4,2,2,2],                           | <- [8,8,2,2]
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16,16,4,4,4],                   | <- [16,16,4,4]
            "use_spectral_norm": false,
            "gin_channels": 256,
            "spk_embed_dim": 109
        }
    }
    """
    config_path = f"configs/{args.sample_rate}.json"
    config_save_path = os.path.join(experiment_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    # Merge - Merge (or override) config from file with arguments
    hparams = HParams(**config)
    hparams.model_dir = hparams.experiment_dir = experiment_dir
    hparams.save_every_epoch     = args.save_every_epoch
    hparams.name                 = args.experiment_dir
    hparams.total_epoch          = args.total_epoch
    hparams.pretrainG            = args.pretrainG
    hparams.pretrainD            = args.pretrainD
    hparams.gpus                 = args.gpus
    hparams.train.batch_size     = args.batch_size # Override
    hparams.sample_rate          = args.sample_rate
    hparams.if_f0                = args.if_f0
    hparams.if_latest            = args.if_latest
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    # Path to the data list file
    hparams.data.training_files  = f"{experiment_dir}/filelist.txt"
    return hparams


def check_git_hash(model_dir):
    """⚡"""
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    """⚡"""
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    """⚡"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
