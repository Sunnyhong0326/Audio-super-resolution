import torch
import torch.nn as nn
from .PixelShuffle1D import PixelShuffle1D
from .AFilm import AFiLM

def init_weights(m):
    if type(m) == nn.Conv1d:
        nn.init.orthogonal_(m.weight)

class AudioUNet(nn.Module):
    """
        self.blocks: number of blocks in downsample/upsample process
        Note: One block consists of multiple layers
        self.n_filters: number of filters for each downsample/upsample block
        self.n_filter_sizes: filter length for each downsample/upsample block
        self.D_blocks: all downsample blocks
        self.B_block: bottleneck block
        self.U_blocks: all upsample blocks
        self.F_block: final block
    """
    def __init__(self,
                 num_blocks = 4,
                 debug = False):
        """
            num_layers: number of blocks in downsample/upsample block
            n_filters: number of filters for each block
            n_filters_sizes: filter length for each block
            up_scale: up scaling factor
        """
        super(AudioUNet, self).__init__()

        self.blocks = num_blocks

        # filter info
        self.n_filters = [65, 33, 17,  9,  9,  9,  9, 9, 9] # default: paper provided 
        self.n_filter_sizes = [128, 256, 512, 512, 512, 512, 512, 512] # default: paper provided

        # self-attention info
        n_step = [4096, 2048, 1024, 512, 256, 512, 1024, 2048, 4096]

        # in out channel info
        in_channel = 1
        out_channel = None

        # downsample blocks 
        self.D_blocks = nn.ModuleList()
        for n_blk in range(self.blocks):
            out_channel = self.n_filters[n_blk]
            if debug:
                print('in_channel:', in_channel, 'out_channel:', out_channel)
            fs = self.n_filter_sizes[n_blk]
            ns = n_step[n_blk]
            nb = int(128 / 2**n_blk) 
            nf = self.n_filters[n_blk]
            D_block = self.downsample_block(in_channel, fs, ns, nb, nf)
            D_block.apply(init_weights)
            self.D_blocks.append(D_block)
            in_channel = out_channel

        # bottleneck block
        out_channel = self.n_filters[-1]
        if debug:
            print('BottleNeck','in_channel:', in_channel, 'out_channel:', out_channel)
        fs = self.n_filter_sizes[-1]
        ns = n_step[self.blocks]
        nb = int(128 / 2**self.blocks) 
        nf = self.n_filters[-1] 
        self.B_block = self.bottleneck_block(in_channel, fs, ns, nb, nf)
        self.B_block.apply(init_weights)

        # upsampling blocks
        self.U_blocks = nn.ModuleList()
        in_channel = out_channel
        for n_blk in reversed(range(self.blocks)):
            out_channel = self.n_filters[n_blk]*2
            if debug:
                print('in_channel:', in_channel, 'out_channel:', out_channel)
            fs = self.n_filter_sizes[n_blk]
            ns = n_step[n_blk] 
            nf = self.n_filters[n_blk]
            U_block = self.upsample_block(in_channel, fs, ns, nb, nf)
            U_block.apply(init_weights)
            self.U_blocks.append(U_block)
            in_channel = out_channel

        # final conv blocks
        self.F_block = self.final_conv(in_channel, 2, 9, 2)
        self.F_block.apply(init_weights)

    def downsample_block(self, in_channel, fs, ns, nb, nf):
        block = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channel, 
                out_channels = nf, 
                kernel_size = fs,
                dilation = 2,
                padding = 'same'),
            nn.MaxPool1d(
                kernel_size = 2
            ),
            nn.LeakyReLU(
                negative_slope = 0.2,
                inplace = True
            ),
            AFiLM(ns, nb, nf)
        )
        return block
    
    def upsample_block(self, in_channel, fs, ns, nb, nf):
        block = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channel,
                out_channels = 2*nf, 
                kernel_size = fs,
                dilation = 2,
                padding = 'same'),
            nn.Dropout(p = 0.5),
            nn.ReLU(inplace = True),
            PixelShuffle1D(2),
            AFiLM(ns, nb, nf)
        )
        return block
    
    def bottleneck_block(self, in_channel, fs, ns, nb, nf):
        block = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channel,
                out_channels = nf,
                kernel_size = fs,
                dilation = 2,
                padding = 'same'
            ),
            nn.MaxPool1d(
                kernel_size = 2
            ),
            nn.Dropout(p = 0.5),
            nn.LeakyReLU(
                negative_slope = 0.2,
                inplace = True
            ),
            AFiLM(ns, nb, nf)
        )
        return block
    

    def final_conv(self, in_channel, out_channel, filter_size, up):
        block = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channel,
                out_channels = out_channel,
                kernel_size = filter_size,
                padding = 'same'
            ),
            PixelShuffle1D(up)
        )
        return block

    

    def forward(self, x):
        # save input for final skip summation
        input_x = x

        down_sample_l = list()
        for n_blk in range(self.blocks):
            x = self.D_blocks[n_blk](x)
            down_sample_l.append(x)

        x = self.B_block(x)

        for n_blk in range(self.blocks):
            x = self.U_blocks[n_blk](x)
            x = torch.cat([x, down_sample_l[self.blocks - n_blk - 1]], dim = 1)

        # final conv propagation
        x = self.F_block(x)
        y = x + input_x
        return y
    
if __name__ == "__main__":
    input = torch.zeros([64,1,8192]).to('cuda')
    model = AudioUNet(debug=True).to('cuda')
    output = model(input)
    print(output.shape)