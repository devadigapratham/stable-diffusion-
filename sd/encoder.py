import torch 
from torch import nn 
from torch.nn import functional as F  
from decoder import VAE_AttentionBlock, VAE_ResidualBlock 

class VAE_Encoder(nn.Sequential): 

    def __init__(self): 
        super().__init__(
          #Following : (Batch_Size, Channels, Height, Width) -> Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3 ,padding=1), 
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), 
            #(Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256), 
            VAE_ResidualBlock(256, 256), 
            #(Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            VAE_ResidualBlock(256, 512), 
            VAE_ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            #(Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),  
            VAE_ResidualBlock(512, 512),         
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512), 
            nn.GroupNorm(32, 512), 
            nn.SiLU(), 
            # Since the padding=1, so width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8). 
            nn.Conv2d(512, 8, kernel_size=3, padding = 1), 
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        ) 
    
    def forward(self, x, noise): 
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)
        for module in self: 
            if getattr(module, 'stride', None) == (2,2): 
                x = F.pad(x, (0, 1, 0, 1)) 

            x = module(x) 
            # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
            mean, log_variance = torch.chunk(x, 2, dim = 1) 
            log_variance = torch.clamp(log_variance, -30, 20) 
            variance = log_variance.exp() 
            stdev = variance.sqrt() 

            x = mean + stdev*noise 
            x *= 0.18215 
            return x 
