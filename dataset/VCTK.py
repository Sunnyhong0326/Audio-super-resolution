import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import librosa
import numpy as np
import h5py
import warnings ; warnings.filterwarnings('ignore')


def read_audio(x_audio_path, y_audio_path, in_sr, out_sr, patch_size):
    '''
    1. Load audio
    2. Crop the audio sample points to args.patch_size to match input of the model
    '''
    # load high and low resolution audio
    audio_lr, _ = librosa.load(x_audio_path, sr = in_sr)
    audio_resample_lr = librosa.resample(audio_lr, orig_sr = in_sr, target_sr = out_sr)
    audio_hr, _ = librosa.load(y_audio_path, sr = out_sr)

    len_audio_hr = len(audio_hr)
    len_resample_lr = len(audio_resample_lr)
    if len_audio_hr > len_resample_lr:
        audio_hr = audio_hr[:len_resample_lr]
        end_num = len_resample_lr - patch_size - 1
    else:
        audio_resample_lr = audio_resample_lr[:len_audio_hr]
        end_num = len_audio_hr - patch_size - 1
    
    if end_num <= 0:
        print('Audio too short:', end_num)
        print('Audio path', x_audio_path)
        
    # random select audio clip
    start_idx = end_num // 2
    end_idx = start_idx + patch_size

    audio_resample_lr = audio_resample_lr[start_idx: end_idx]
    audio_hr = audio_hr[start_idx: end_idx]

    # normalize the amplitude to -1 and 1
    # x_scale = np.max(np.abs(audio_resample_lr))
    # y_scale = np.max(np.abs(audio_hr))
    # audio_resample_lr = audio_resample_lr / x_scale
    # audio_hr = audio_hr / y_scale

    # shape of audio clip (1, 8192)
    audio_resample_lr = np.expand_dims(audio_resample_lr, axis = 0)
    audio_hr = np.expand_dims(audio_hr, axis = 0)

    # convert to tensor
    audio_resample_lr = torch.tensor(audio_resample_lr)
    audio_hr = torch.tensor(audio_hr)
    return audio_resample_lr, audio_hr

class VCTKData(Dataset):
    def __init__(self, args, h5_filename):
        '''
        self.audio_x_path: input low resolution audio directory path
        self.audio_y_path: target high resolution audio directory path
        '''
        self.h5_path = os.path.join(args.audio_path, h5_filename)

        with h5py.File(self.h5_path, 'r') as f:
            X = f['data'].shape
            Y = f['label'].shape
            assert X == Y
            self.length = X[0]

        # audio_dir = Path(audio_dataset_path)
        # x_dir_path = audio_dir / '16k'
        # y_dir_path = audio_dir / '48k'

        # self.audio_x_path = list()
        # self.audio_y_path = list()
        # self.in_sr = args.in_sr
        # self.out_sr = args.out_sr
        # self.patch_size = args.patch_size

        # # load input audio path
        # for path in x_dir_path.iterdir():
        #     if path.is_dir():
        #         for audio in path.iterdir():
        #             if audio.suffix == '.wav':
        #                 self.audio_x_path.append(audio)

        # # load target audio path
        # for path in y_dir_path.iterdir():
        #     if path.is_dir():
        #         for audio in path.iterdir():
        #             if audio.suffix == '.wav':
        #                 self.audio_y_path.append(audio)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # audio_lr, audio_hr = read_audio(
        #     self.audio_x_path[idx], 
        #     self.audio_y_path[idx],
        #     self.in_sr,
        #     self.out_sr,
        #     self.patch_size
        #     )
        with h5py.File(self.h5_path, 'r') as hf:
            audio_lr = torch.Tensor(hf['data'][idx])
            audio_hr = torch.Tensor(hf['label'][idx])

        return {'lr': audio_lr, 'hr': audio_hr}

