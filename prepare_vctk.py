import os
import numpy as np
import librosa
import h5py
from scipy import interpolate
from scipy.signal import decimate
def upsample(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    return x_sp

def create(dataset_name, X, Y):
    with h5py.File(dataset_name, 'w') as f:
        f.create_dataset('data', data=X, maxshape=(None,1,8192), dtype=np.float32) # lr
        f.create_dataset('label', data=Y, maxshape=(None,1,8192), dtype=np.float32) # hr
    print(f'create complete -> {dataset_name}')


def append(dataset_name, new_data, new_label):
    with h5py.File(dataset_name, 'a') as f:
        f['data'].resize((f['data'].shape[0] + new_data.shape[0]), axis=0)
        f['data'][-new_data.shape[0]:] = new_data

        f['label'].resize((f['label'].shape[0] + new_label.shape[0]), axis=0)
        f['label'][-new_label.shape[0]:] = new_label
    print(f'append complete -> {dataset_name}')

def preprocess2(hr_file_list, filename, hr_sr = 48000, patch_size = 8192, stride = 4096):
    dataset_name = os.path.join('./data', filename)
    scale = 4
    for i, hr_wav in enumerate(hr_file_list):
        hr_patches = list()
        lr_patches = list()
        x, fs = librosa.load(hr_wav, sr=hr_sr)

        # crop so that it works with scaling ratio
        x_len = len(x)
        x = x[ : x_len - (x_len % 4)]

        # generate low-res version
        x_lr = decimate(x, scale)
        x_lr = upsample(x_lr, scale)
        assert len(x) % scale == 0
        assert len(x_lr) == len(x)

        # generate patches
        max_i = len(x) - patch_size + 1
        for i in range(0, max_i, stride):
            # keep only a fraction of all the patches
            u = np.random.uniform()
            if u > 1: continue
            i_lr = i

            hr_patch = np.array( x[i : i+patch_size] )
            lr_patch = np.array( x_lr[i_lr : i_lr+patch_size] )

            assert len(hr_patch) == patch_size
            assert len(lr_patch) == patch_size

            hr_patches.append(hr_patch.reshape((1, patch_size)))
            lr_patches.append(lr_patch.reshape((1, patch_size)))
            
        lr_patches = np.array(lr_patches)
        hr_patches = np.array(hr_patches)
        if i == 0:
            create(dataset_name, lr_patches, hr_patches)
            append(dataset_name, lr_patches, hr_patches)
        else:
            append(dataset_name, lr_patches, hr_patches)


def preprocess(lr_file_list, hr_file_list, filename, lr_sr=12000, hr_sr=48000, patch_size=8192, stride=16384):
    dataset_name = os.path.join('./data', filename)
    scale = hr_sr // lr_sr

    for idx, lr_wav, hr_wav in zip(range(len(lr_file_list)),lr_file_list, hr_file_list):
        lr_patches = list()
        hr_patches = list()
        
        x_lr, _ = librosa.load(lr_wav, sr=lr_sr)
        x_hr, _ = librosa.load(hr_wav, sr=hr_sr)

        x_lr = upsample(x_lr, scale)

        len_lr = len(x_lr)
        len_hr = len(x_hr)

        total_len = len_lr if len_lr < len_hr else len_hr

        if len_lr > len_hr:
            x_lr = x_lr[:len_hr-len_hr%patch_size]
            x_hr = x_hr[:len_hr-len_hr%patch_size]
            total_len = len_hr-len_hr%patch_size
        else:
            x_lr = x_lr[:len_lr-len_lr%patch_size]
            x_hr = x_hr[:len_lr-len_lr%patch_size]
            total_len = len_lr-len_lr%patch_size

        for i in range(0, total_len-patch_size, stride):
            lr_patch = x_lr[i:i+patch_size]
            hr_patch = x_hr[i:i+patch_size]

            lr_patch = np.expand_dims(lr_patch, axis = 0)
            hr_patch = np.expand_dims(hr_patch, axis = 0)

            assert lr_patch.shape == (1,8192)
            assert hr_patch.shape == (1,8192)

            lr_patches.append(lr_patch)
            hr_patches.append(hr_patch)

        lr_patches = np.array(lr_patches)
        hr_patches = np.array(hr_patches)

        if idx == 0:
            create(dataset_name, lr_patches, hr_patches)
            append(dataset_name, lr_patches, hr_patches)
        else:
            append(dataset_name, lr_patches, hr_patches)

    print('Finish creating dataset')
    
    return lr_patches, hr_patches, dataset_name

def load_wav_list(dirname, num_speakers = 10, num_files = 20):
    file_list = []
    dirname = dirname
    filenames = os.listdir(dirname)
    for filename in sorted(filenames)[:num_speakers]:
        full_filename = os.path.join(dirname, filename)
        for files in sorted(os.listdir(full_filename)): 
            file_list.append(os.path.join(full_filename,files))
    print('load wav list examples..')
    print('length' , len(file_list))
    for i, file in enumerate(file_list):
        print(file)
        if i > 5: break

    return file_list


def prepare_vctk():
    #single_train_lr_file_list = load_wav_list('./data/train/train/16k', num_speakers = 1, num_files = -1)
    single_train_hr_file_list = load_wav_list('./data/train/train/48k', num_speakers = 1, num_files = -1 )

    #single_val_lr_file_list = load_wav_list('./data/val/val/16k', num_speakers = 1, num_files = -1)
    single_val_hr_file_list = load_wav_list('./data/val/val/48k', num_speakers = 1, num_files = -1)

    preprocess2(single_train_hr_file_list, filename='Single_Train_Dataset.h5')
    preprocess2(single_val_hr_file_list, filename='Single_Valid_Dataset.h5')

    # multiple_train_lr_file_list = load_wav_list('./data/train/train/16k', num_speakers = -1, num_files = 20)
    # multiple_train_hr_file_list = load_wav_list('./data/train/train/48k', num_speakers = -1, num_files = 20)

    # multiple_val_lr_file_list = load_wav_list('./data/val/val/16k', num_speakers= 10, num_files = 20)
    # multiple_val_hr_file_list = load_wav_list('./data/val/val/48k', num_speakers= 10, num_files = 20)

    # preprocess(multiple_train_lr_file_list, multiple_train_hr_file_list, filename='Multiple_Train_Dataset.h5', stride=4096)
    # preprocess(multiple_val_lr_file_list, multiple_val_hr_file_list, filename='Multiple_Valid_Dataset.h5')
    

if __name__ == "__main__":
    prepare_vctk()