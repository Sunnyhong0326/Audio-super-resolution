import argparse

def make_train_parser():
    parser = argparse.ArgumentParser()
    
    # path of experiment results
    parser.add_argument('--result_path', type = str, default = './results', help = 'the result path')
    parser.add_argument('--exp' ,type = str, required = True, help = 'experiment name')

    # hyperparameter info
    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
    parser.add_argument('--num_epochs', type = int, default = 50, help = 'number of epochs')
    parser.add_argument('--lr', type = float, default = 3e-4, help = 'learning rate')
    
    # model info
    parser.add_argument('--num_blocks', type = int, default = 4, help = 'number of downsample or upsample blocks')
    parser.add_argument('--patch_size', type = int, default = 8192, help = "input length that can feed into model")
    parser.add_argument('--resume_train', type=bool, default=False, help='Resume Training')
    parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained model')
    parser.add_argument('--ckpt', type=str, default='EXP', help='load from checkpoint')

    # dataset info
    parser.add_argument('--audio_path', type = str, default = './data', help = 'dataset path')
    parser.add_argument('--in_sr', type = int, default = 12000, help = "input sample rate")
    parser.add_argument('--out_sr', type = int, default = 48000, help = "output sample rate")
    parser.add_argument('--up_scale',  type = int, default = 4, help = "up scaling factor")
    parser.add_argument('--dataset', type = str, default='Single', help='dataset type (can be multiple speaker or single speaker)')

    hparams = parser.parse_args()
    return hparams   
    
def make_test_parser():
    parser = argparse.ArgumentParser()

    # testing dataset info
    parser.add_argument('--audio_path', type = str, help = 'the path of audio that we want to test')
    parser.add_argument('--in_sr', type = int, default = 12000, help = "input sample rate")
    parser.add_argument('--out_sr', type = int, default = 48000, help = "output sample rate")
    parser.add_argument('--up_scale',  type = int, default = 4, help = "up scaling factor")

    # path of testing experiment result
    parser.add_argument('--exp', type = str, default = 'First',help = 'experiment name')
    parser.add_argument('--log_path', type = str, default = 'logs', help='log path')
    
    # model info
    parser.add_argument('--batch_size', type = int, default = 64, help='batch_size')
    parser.add_argument('--ckpt', type = str, required = True, help='pretrained model check point path')
    
    hparams = parser.parse_args()
    return hparams
    
if __name__== '__main__':
    #train_hparams = make_train_parser()
    test_hparams = make_train_parser()
    #print('train parser:', train_hparams)
    print('train parser:', test_hparams)