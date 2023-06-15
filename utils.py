import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('Agg')
from models.AudioUNet import AudioUNet
import torch
from scipy import interpolate
from scipy.signal import decimate

#origin paper provide spectrugram
def get_spectrum(data, n_fft=2048):
    data=data[0:len(data)-len(data)%n_fft]
    S = librosa.stft(data, n_fft=2048)
    S = np.log1p(np.abs(S))
    return S


def plot_train_and_gt_spectrum(train, gt, destfile, num_epochs):
    train_spectrum = get_spectrum(train)
    gt_spectrum = get_spectrum(gt)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.tight_layout()
    fig.dpi = 100
    axes[0].imshow(train_spectrum.T, aspect = 10)
    axes[0].set_title(f'Predicted spectrogram')
    axes[1].imshow(gt_spectrum.T, aspect = 10)
    axes[1].set_title(f'Ground truth spectrogram')
    plt.savefig(destfile + f'Epoch{num_epochs}'+'_spectogram.png')
    plt.show()


def plot_curve(training_result_dir, train, valid, epoch, title):
    # Plot the loss curve against epoch
    file_name = f'{title} Curve.jpg'
    plt.figure(dpi = 100)
    plt.title(title)
    plt.plot(range(epoch), train, label='Train')
    plt.plot(range(epoch), valid, label='Valid')
    plt.legend()
    plt.savefig(os.path.join(training_result_dir, file_name))
    plt.close()

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)
  x_sp = interpolate.splev(i_hr, f)

  return x_sp

def up_sample_wav_12_to_48(
      test_audio_path,
      ckpt,
      in_sr = 12000,
      r = 4,
      ):
    '''
    read audio, normalized to -1 and 1 and crop the audio sample points to multiple of args.patch_size
    '''
    # check device if CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model and hyper parameters
    checkpoint = torch.load(ckpt)
    model = AudioUNet().to(device)
    num_blocks = 4
    patch_size = checkpoint['patch_size']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # preprocess the low resolution audio to be multiple of model patch_size
    audio_hr, _ = librosa.load(test_audio_path, sr = in_sr)
    #print(audio_lr.shape)
    audio_hr  = np.pad(audio_hr, (0, patch_size - (audio_hr.shape[0] % patch_size)), 'constant', constant_values=(0,0))
    #print(audio_lr.shape)
    audio_lr = decimate(audio_hr, 4)  
    
    # normalize the amplitude to -1 and 1 and reshape to (1, 1, len(audio))
    # x_scale = np.max(np.abs(audio_lr))
    # audio_lr = audio_lr / x_scale
    audio_lr = audio_lr.reshape((1, 1, len(audio_lr)))

    # preprocessing
    # assert len(audio_lr) == 1
    # x_sp = spline_up(audio_lr, r)
    # x_sp = x_sp[: len(x_sp) - (len(x_sp) % (2**(num_blocks+1)))]
    # x_sp = x_sp.reshape((1 , 1, len(x_sp)))
    # # reshape to (batch, 1, 1892) and change to tensor
    # x_sp = x_sp.reshape((int(x_sp.shape[2]/patch_size), 1, patch_size))
    # x_sp = torch.Tensor(x_sp).cuda()
    #print(x_sp.shape)

    model.eval()
    with torch.no_grad():
      predict_hr = model(audio_lr)
    #print(predict_hr.shape)
    predict_hr = predict_hr.flatten()
    predict_hr = predict_hr.detach().cpu().numpy()
    #print(predict_hr.shape)
    
    return predict_hr
