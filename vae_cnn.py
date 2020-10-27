"""
Variational AutoEncoder framework
Based on CNN net
Modification:
- 14/3/2019: Implementing Lucas RCNN5 model
# Author: cain@cinnamon.is
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """"
    Encoder module
    """
    def __init__(self, input_dim=[64, 64],
                        hidden_size=None,
                        latent_size=2,
                        drop_rate=0.5,
                        use_cuda = False):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_size = latent_size
        self.hidden_len = 512*(input_dim[0]//32-1)*((input_dim[1]//4-1)//2-3)
        self.max_pool_idx = {}

        if hidden_size is None:
            self.hidden_size = self.hidden_len
        else:
            self.hidden_size = 512*hidden_size
        self.use_cuda = use_cuda
        ## Definition encoder
        self.conv_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2) #conv1
        self.max_pool_1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.rcl1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # RCL 1
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.max_pool_rcl1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.drop_rcl1 = nn.Dropout(drop_rate)

        self.rcl2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # RCL 2
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.max_pool_rcl2 = nn.MaxPool2d((2, 2), (2, 1), (0, 0), return_indices=True)
        self.drop_rcl2 = nn.Dropout(drop_rate)

        self.rcl3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # RCL 3
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.max_pool_rcl3 = nn.MaxPool2d(2, 2, return_indices=True)
        self.drop_rcl3 = nn.Dropout(drop_rate)

        self.conv_last = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(0, 0)),  # Final Conv
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.max_pool_last = nn.MaxPool2d((2, 2), (2, 1), (0, 0), return_indices=True)
        """
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim[0]*self.input_dim[1], self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )"""

        #self.mean = nn.Conv1d(self.hidden_size, self.latent_size, (1, 1), stride=1, padding=0)
        self.mean = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar = nn.Linear(self.hidden_size, self.latent_size)
        #self.logvar = nn.Conv1d(self.hidden_size, self.latent_size, (1, 1), stride=1, padding=0)

    def _sampling(self, latent_dist):
        mean = latent_dist["mean"]
        std = latent_dist["logvar"].mul(0.5).exp()
        if self.training:
            eps = torch.randn_like(std)
            if self.use_cuda:
                eps = eps.cuda()
            return eps.mul(std).add_(mean)
        else:
            return mean

    def forward(self, x):
        ## Encoding
        h = self.conv_1(x)
        h, self.max_pool_idx["conv_1"] = self.max_pool_1(h)

        h = self.rcl1(h)
        h, self.max_pool_idx["rcl1"] = self.max_pool_rcl1(h)
        h = self.drop_rcl1(h)

        h = self.rcl2(h)
        h, self.max_pool_idx["rcl2"] = self.max_pool_rcl2(h)
        h = self.drop_rcl2(h)

        h = self.rcl3(h)
        h, self.max_pool_idx["rcl3"] = self.max_pool_rcl3(h)
        h = self.drop_rcl3(h)

        h = self.conv_last(h)
        h, self.max_pool_idx["conv_last"] = self.max_pool_last(h)

        #h = self.encoder(x.view(-1, self.input_dim[0]*self.input_dim[1]))
        h = h.transpose(3, 1).contiguous()
        latent_dist = {"mean": self.mean(h.view(-1, self.hidden_size)),#self.mean(h.view(-1, self.hidden_size, 1, self.hidden_len//self.hidden_size)),
                        "logvar": self.logvar(h.view(-1, self.hidden_size))}#self.logvar(h.view(-1, self.hidden_size, 1, self.hidden_len//self.hidden_size))}
        enc_output = self._sampling(latent_dist)
        return enc_output, latent_dist

class Decoder(nn.Module):
    """"
    Decoder module
    """
    def __init__(self, input_dim=[64, 64],
                        hidden_size=None,
                        latent_size=2):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_size = latent_size
        self.hidden_len = 512*(input_dim[0]//32-1)*((input_dim[1]//4-1)//2-3)
        if hidden_size is None:
            self.hidden_size = self.hidden_len
        else:
            self.hidden_size = 512* hidden_size

        #self.h = nn.ConvTranspose1d(self.latent_size, self.hidden_size, (1, 1), stride=1, padding=0)
        self.h = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.ReLU()
        )
        ## Definition decoder
        self.inv_conv_last = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (3, 3), stride=1, padding=0), ## Reverse out layer
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.max_unpool_last = nn.MaxUnpool2d((2, 2), (2, 1), (0, 0))

        self.inv_rcl3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3), stride=1, padding=1),  ## Reverse RCL 3
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.max_unpool_rcl3 = nn.MaxUnpool2d((2, 2))

        self.inv_rcl2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), stride=1, padding=1),  ## Reverse RCL 2
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.max_unpool_rcl2 = nn.MaxUnpool2d((2, 2), (2, 1), (0, 0))

        self.inv_rcl1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (3, 3), stride=1, padding=1),  ## Reverse RCL 1
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.max_unpool_rcl1 = nn.MaxUnpool2d((2, 2)) ### W%4

        self.inv_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, (5, 5), stride=1, padding=2),  ## Reverse first conv
            nn.Sigmoid()
        )
        self.max_unpool_conv_1 = nn.MaxUnpool2d((2, 2)) ### W%8



    def forward(self, z, max_pool_idx, batch_size=16):
        h = self.h(z)
        h = h.view(batch_size, -1, self.input_dim[0] // 32 - 1, 512).transpose(1, 3)  # 10, 178, 1, 512

        h = self.max_unpool_last(h, max_pool_idx["conv_last"])
        h = self.inv_conv_last(h)

        h = self.max_unpool_rcl3(h, max_pool_idx["rcl3"])
        h = self.inv_rcl3(h)

        h = self.max_unpool_rcl2(h, max_pool_idx["rcl2"])
        h = self.inv_rcl2(h)

        h = self.max_unpool_rcl1(h, max_pool_idx["rcl1"])
        h = self.inv_rcl1(h)

        h = self.max_unpool_conv_1(h, max_pool_idx["conv_1"])
        out = self.inv_conv_1(h)

        return out
        #return self.decoder(h).reshape(-1,1,self.input_dim[0],self.input_dim[1])