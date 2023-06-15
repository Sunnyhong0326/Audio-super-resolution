import numpy as np
import librosa

def AvgPSNR(predict_hr, gt_hr):
    num_clips = len(predict_hr)
    avg_psnr = 0
    for idx in range(num_clips):
        psnr = PSNR(predict_hr[idx], gt_hr[idx])
        avg_psnr += psnr / num_clips
    return avg_psnr

def PSNR(one_predict_hr, one_gt_hr):
    mse = np.mean((np.array(one_gt_hr, dtype=np.float32) - np.array(one_predict_hr, dtype=np.float32)) ** 2)
    return 20 * np.log10(np.max(one_gt_hr) / (np.sqrt(mse)))

def AvgLSD(predict_hr, gt_hr):
    num_clips = len(predict_hr)
    avg_lsd = 0
    for idx in range(num_clips):
        lsd = LSD(predict_hr[idx], gt_hr[idx])
        avg_lsd += lsd / num_clips
    return avg_lsd

def LSD(one_predict_hr, one_gt_hr):
    spectrogram1 = np.abs(librosa.stft(one_predict_hr))
    spectrogram2 = np.abs(librosa.stft(one_gt_hr))
    spectrogram_log1 = np.log10(spectrogram1 ** 2)
    spectrogram_log2 = np.log10(spectrogram2 ** 2)
    original_target_squared = (spectrogram_log1 - spectrogram_log2) ** 2
    log_spectral_distance = np.mean(np.sqrt(np.mean(original_target_squared, axis = 0)))
    return log_spectral_distance
