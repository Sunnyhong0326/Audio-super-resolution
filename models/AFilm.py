import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000, device = 'cuda'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model).to(device)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        #self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, channel, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads=8, ff_dim=2048, rate=0.1, device = 'cuda'):
        super(TransformerEncoder, self).__init__()

        self.pe = PositionalEncoding(d_model=embed_dim, device = device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=rate).to(device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(device)
        self.dropout = nn.Dropout(rate)

    def forward(self, x):
        # adding embedding and position encoding.
        x *= torch.sqrt(torch.Tensor([32]).to('cuda'))
        x = self.pe(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        
        return x  # (batch_size, d_model, input_seq_len)

    
class AFiLM(nn.Module):
    def __init__(self, n_step, block_size, n_filters):
        """
            n_step: T
            block_size: block size
            n_filters: channel size
        """
        super(AFiLM, self).__init__()
        self.block_size = block_size
        self.n_filters = n_filters
        self.n_step = n_step
        self.n_blocks = int(self.n_step/self.block_size)

        self.maxPool = nn.MaxPool1d(
            kernel_size = self.block_size,
            stride = self.block_size
            )
        self.transformer = TransformerEncoder(embed_dim=self.n_blocks,
                                              num_layers=4)

    def make_normalizer(self, x_in):
        ## (B, C, T)
        x_in_down = self.maxPool(x_in)
        # (B, C, X) X: T/blocksize
        x_transformer = self.transformer(x_in_down).to('cuda')
        # (B, C, X)
        return x_transformer

    def apply_normalizer(self, x_in, x_norm):
        x_norm = x_norm.view([-1, self.n_filters, 1, self.n_blocks])
        x_in = x_in.view([-1, self.n_filters, self.block_size, self.n_blocks])
        x_out = x_norm * x_in
        x_out = x_out.view([-1, self.n_filters, self.n_blocks*self.block_size])
        return x_out
    
    def forward(self, x):
        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x

if __name__ == "__main__":
    input_tensor = torch.zeros([1, 2, 4096]).to('cuda')
    model = AFiLM(4096, 128, 2)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)