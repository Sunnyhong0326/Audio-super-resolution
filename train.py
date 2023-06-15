from opts import make_train_parser
from tqdm import tqdm
import os
from dataset.VCTK import VCTKData

from models.AudioUNet import AudioUNet
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from metrics import AvgPSNR, AvgLSD
import torch.nn.functional as F
from losses.stftloss import MultiResolutionSTFTLoss
from utils import plot_curve

# seed  
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(99)

def train(hparams):
    train_data = VCTKData(hparams, h5_filename=hparams.dataset+'_Train_Dataset.h5')
    val_data = VCTKData(hparams, h5_filename=hparams.dataset+'_Valid_Dataset.h5')
    
    train_loader = DataLoader(train_data, batch_size = hparams.batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    val_loader = DataLoader(val_data, batch_size = hparams.batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    
    model = AudioUNet(hparams.num_blocks)
    optim = Adam(model.parameters(), lr = hparams.lr, betas=(0.9, 0.999))
    #scheduler = CosineAnnealingLR(optimizer = optim, T_max = start_epoch + hparams.num_epochs, eta_min = hparams.lr*0.01)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    mrstftloss = MultiResolutionSTFTLoss().to(device)

    result_path = os.path.join(hparams.result_path, hparams.exp)
    metrics = {'train_loss': [], 'valid_loss': [], 'train_psnr':[], 'valid_psnr':[], 'train_lsd':[], 'valid_lsd':[]}

    if hparams.resume_train:
        ckpt_path = f'ckpts/{hparams.exp}.pth'
        checkpoint = torch.load(ckpt_path)
        optim.load_state_dict(checkpoint['optimizer_state_dict']) 
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics = checkpoint['metrics']
        start_epoch = checkpoint['epoch']
    elif hparams.pretrained:
        ckpt_path = f'ckpts/{hparams.ckpt}.pth'
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = 0
    else:
        start_epoch = 0

    with tqdm(hparams.num_epochs) as pbar:
        for i_epoch in range(1+start_epoch, 1+start_epoch + hparams.num_epochs):
            # train
            #############################################
            total_train_loss = 0
            predict_train_result = []
            gt_hr_train = []
            model.train()
            for data in iter(train_loader):
                batch_train_lr = data['lr'].to(device)
                batch_train_hr = data['hr'].to(device)
                # forward propagation
                predict_train_hr = model(batch_train_lr)
                # compute loss
                with torch.autograd.set_detect_anomaly(True):
                    loss = F.mse_loss(batch_train_hr, predict_train_hr)
                    sc_loss, mag_loss = mrstftloss(predict_train_hr.squeeze(1), batch_train_hr.squeeze(1))
                    loss += sc_loss + mag_loss
                # clear grad buffer
                model.zero_grad()
                # backward to compute gradient
                loss.backward()
                # update model's weight
                optim.step()
                # accumulate loss
                total_train_loss += loss.detach().item()
                del loss
                predict_train_result.append(predict_train_hr.detach().cpu().numpy())
                gt_hr_train.append(batch_train_hr.detach().cpu().numpy())

            avg_train_loss = total_train_loss / len(train_data)
            avg_train_psnr = AvgPSNR(predict_train_result, gt_hr_train)
            avg_train_lsd = AvgLSD(predict_train_result, gt_hr_train)
            metrics['train_loss'].append(avg_train_loss)
            metrics['train_psnr'].append(avg_train_psnr)
            metrics['train_lsd'].append(avg_train_lsd)
            #############################################

            #################################################
            # validation
            model.eval()
            total_val_loss = 0
            predict_val_result = []
            gt_hr_val = []

            with torch.no_grad():
                for data in iter(val_loader):
                    batch_val_lr = data['lr'].to(device)
                    batch_val_hr = data['hr'].to(device)
                    predict_val_hr = model(batch_val_lr)
                    # compute loss
                    with torch.autograd.set_detect_anomaly(True):
                        loss = F.mse_loss(batch_train_hr, predict_train_hr)
                        sc_loss, mag_loss = mrstftloss(predict_train_hr.squeeze(1), batch_train_hr.squeeze(1))
                        loss += sc_loss + mag_loss
                    total_val_loss += loss.detach().item()
                    # Save predicted high resolution audio and actual high resolution audio 
                    # and convert to numpy from GPU to CPU
                    predict_val_result.append(predict_val_hr.detach().cpu().numpy())
                    gt_hr_val.append(batch_val_hr.detach().cpu().numpy())

            avg_val_loss = total_val_loss / len(val_data)
            avg_val_psnr = AvgPSNR(predict_val_result, gt_hr_val)
            avg_val_lsd = AvgLSD(predict_val_result, gt_hr_val)
            metrics['valid_loss'].append(avg_val_loss)
            metrics['valid_psnr'].append(avg_val_psnr)
            metrics['valid_lsd'].append(avg_val_lsd)
            #####################################################################
            #scheduler.step()

            # update my process
            pbar.set_description(f'Epoch [{i_epoch}/{start_epoch + hparams.num_epochs}]') # prefix str
            
            # use pbar.set_postfix() to setting infomation : train_loss , val_loss , val_psnr and val_LSD
            pbar.set_postfix(
                Avg_Train_Loss = avg_train_loss,
                Avg_Val_Loss = avg_val_loss,
                )
            pbar.update(1)
            
            #plot_train_and_gt_spectrum(predict_val_result[0][0], gt_hr_val[0][0], os.path.join(hparams.result_path, hparams.exp), i_epoch)
            #up_sample_wav_12_to_48(test_audio_path, )
            # saving the checkpoint   
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'epoch': i_epoch,
                'metrics': metrics,
                'patch_size': hparams.patch_size,
                },
                f'ckpts/{hparams.exp}.pth')
            plot_curve(result_path, metrics['train_loss'], metrics['valid_loss'], i_epoch, 'Loss')
            plot_curve(result_path, metrics['train_psnr'], metrics['valid_psnr'], i_epoch, 'Avg PSNR')
            plot_curve(result_path, metrics['train_lsd'], metrics['valid_lsd'], i_epoch, 'Avg LSD')

    
if __name__=='__main__':
    hparams = make_train_parser()
    print(hparams)
    print('cuda is available', torch.cuda.is_available())
    if os.path.isdir(os.path.join(hparams.result_path, hparams.exp)) == False:
        os.makedirs(os.path.join(hparams.result_path, hparams.exp))
    train(hparams)