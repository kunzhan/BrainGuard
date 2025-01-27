import torch
import torch.nn as nn
from functools import partial
import math


from diffusers.models.vae import Decoder
class Voxel2StableDiffusionModel(torch.nn.Module):
    def __init__(self, in_dim=15724, h=4096, n_blocks=4, use_cont=False, ups_mode='4x'):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25)
            ) for _ in range(n_blocks)
        ])
        self.ups_mode = ups_mode
        if ups_mode=='4x':
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 64)
            
            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()
        
        if ups_mode=='8x':  # prev best
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)
            
            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()
        
        if ups_mode=='16x':
            self.lin1 = nn.Linear(h, 8192, bias=False)
            self.norm = nn.GroupNorm(1, 512)
            
            self.upsampler = Decoder(
                in_channels=512,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256, 512],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()

    # @torchsnooper.snoop()
    def forward(self, x, return_transformer_feats=False):
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # bs, 4096

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        if self.ups_mode == '16x':
            side = 4
        
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous())
        if return_transformer_feats:
            return self.upsampler(x), self.maps_projector(x).flatten(2).permute(0,2,1)
        return self.upsampler(x)


class BrainGuardModule(nn.Module):
    def __init__(self):
        super(BrainGuardModule, self).__init__()
    def forward(self, x):
        return x
    
class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_size, out_features): 
        super(RidgeRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, out_features)
    def forward(self, x):
        x = self.linear(x)
        return x

class BrainNetwork(nn.Module):
  def __init__(self, out_dim_image=768, out_dim_text=768, in_dim=15724, latent_size=768, h=2048, n_blocks=4, norm_type='ln', use_projector=True, act_first=False, drop1=.5, drop2=.15, train_type='vision'):
    super().__init__()
    norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
    act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
    act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
    self.mlp = nn.ModuleList([
        nn.Sequential(
            nn.Linear(h, h),
            *[item() for item in act_and_norm],
            nn.Dropout(drop2)
        ) for _ in range(n_blocks)
    ])
    self.head_image = nn.Linear(h, out_dim_image, bias=True)
    self.head_text = nn.Linear(h, out_dim_text, bias=True)
    self.n_blocks = n_blocks
    self.latent_size = latent_size
    self.use_projector = use_projector
    self.train_type = train_type
    if use_projector:
        self.projector_image = nn.Sequential(
        nn.LayerNorm(self.latent_size),
        nn.GELU(),
        nn.Linear(self.latent_size, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Linear(2048, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Linear(2048, self.latent_size)
)
        self.projector_text = nn.Sequential(
        nn.LayerNorm(self.latent_size),
        nn.GELU(),
        nn.Linear(self.latent_size, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Linear(2048, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Linear(2048, self.latent_size)
)
        
  def forward(self, x):
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x_image = self.head_image(x) 
        x_text = self.head_text(x)
        if self.use_projector: 
            return self.projector_image(x_image.reshape(len(x_image), -1, self.latent_size)), self.projector_text(x_text.reshape(len(x_text), -1, self.latent_size))  
        return x
