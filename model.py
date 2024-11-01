

import torch
import torch.nn as nn
import math

# Bloque de Atención de Canal con regularización L2
class ChannelAttention(nn.Module):
    def __init__(self, n_feats, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Regularización L2 aplicada en cada convolución del bloque de atención
        self.fc1 = nn.Conv3d(n_feats, n_feats // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(n_feats // ratio, n_feats, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x

# Bloque CBAM simplificado (solo canal)
class CBAM(nn.Module):
    def __init__(self, n_feats, ratio=8):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(n_feats, ratio)

    def forward(self, x):
        x = self.channel_attention(x)
        return x

# Clase TwoCNN con regularización y CBAM simplificado
class TwoCNN(nn.Module):
    def __init__(self, wn, n_feats=64): 
        super(TwoCNN, self).__init__()
        self.body = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3), stride=1, padding=(1,1), bias=False))
        
        # Bloque CBAM simplificado
        self.cbam = CBAM(n_feats)
        
        # Capa de ajuste de canales, se configura dinámicamente en forward
        self.adjust_channels = None

    def forward(self, x):
        out = self.body(x)
        
        # Pasamos `out` a través del CBAM y ajustamos los canales si es necesario
        out = self.cbam(out.unsqueeze(2)).squeeze(2)  # Adaptamos CBAM a 2D convolución

        # Configuración dinámica de `adjust_channels` para igualar canales de `x`
        if out.shape[1] != x.shape[1]:
            self.adjust_channels = nn.Conv2d(out.shape[1], x.shape[1], kernel_size=1).to(out.device)
            out = self.adjust_channels(out)
        
        # Realizamos la suma
        out = torch.add(out, x)
        return out             

# Clase ThreeCNN con regularización y CBAM simplificado
class ThreeCNN(nn.Module):
    def __init__(self, wn, n_feats=64):
        super(ThreeCNN, self).__init__()
        self.act = nn.ReLU(inplace=True)

        # Inicializamos las capas de convolución 3D con n_feats canales
        body_spatial = []
        for i in range(2):
            body_spatial.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)))
        
        body_spectral = []
        for i in range(2):
            body_spectral.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)))
        
        self.body_spatial = nn.Sequential(*body_spatial)
        self.body_spectral = nn.Sequential(*body_spectral)
        
        # Bloque CBAM simplificado
        self.cbam = CBAM(n_feats)
        
        # Ajuste de canales: se configura dinámicamente en forward() según el tamaño de entrada
        self.adjust_channels = None

    def forward(self, x): 
        out = x
        for i in range(2):  
            out_spatial = self.body_spatial[i](out)
            out_spectral = self.body_spectral[i](out)
            
            # Sumamos out_spatial y out_spectral
            out = torch.add(out_spatial, out_spectral)
            if i == 0:
                out = self.act(out)
        
        out = self.cbam(out)
        
        # Configuración dinámica de self.adjust_channels para igualar canales de `x`
        if out.shape[1] != x.shape[1]:
            self.adjust_channels = nn.Conv3d(out.shape[1], x.shape[1], kernel_size=1).to(out.device)
            out = self.adjust_channels(out)
        
        # Verificamos la compatibilidad de las dimensiones antes de la suma final con x
        if out.shape == x.shape:
            out = torch.add(out, x)
        else:
            print(f"Dimension mismatch before final addition: out {out.shape}, x {x.shape}")
            return None
        
        return out

# Clase SFCCBAM
class SFCCBAM(nn.Module):
    def __init__(self, args):
        super(SFCCBAM, self).__init__()
        
        scale = args.upscale_factor
        n_feats = args.n_feats
        self.n_module = args.n_module        
                 
        wn = lambda x: torch.nn.utils.weight_norm(x)
    
        self.gamma_X = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_Y = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_DFF = nn.Parameter(torch.ones(4))
        self.gamma_FCF = nn.Parameter(torch.ones(2))
        
        # Head 
        ThreeHead = [wn(nn.Conv3d(1, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)),
                     wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))]
        self.ThreeHead = nn.Sequential(*ThreeHead)

        TwoHead = [wn(nn.Conv2d(1, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False))]
        self.TwoHead = nn.Sequential(*TwoHead)

        # Tail 
        TwoTail = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*4, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)))
                TwoTail.append(nn.PixelShuffle(2))           
        else:
            TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*9, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)))
            TwoTail.append(nn.PixelShuffle(3))  
        TwoTail.append(wn(nn.Conv2d(n_feats, 1, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False)))                                 	    	
        self.TwoTail = nn.Sequential(*TwoTail)

        # Convoluciones y atenciones
        self.twoCNN = nn.Sequential(*[TwoCNN(wn, n_feats) for _ in range(self.n_module)])
        self.reduceD_Y = wn(nn.Conv2d(n_feats*self.n_module, n_feats, kernel_size=(1,1), stride=1, bias=False))                          
        self.twofusion = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False))

        self.threeCNN = nn.Sequential(*[ThreeCNN(wn, n_feats) for _ in range(self.n_module)])
        self.reduceD = nn.Sequential(*[wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1, bias=False)) for _ in range(self.n_module)])                              
        self.reduceD_X = wn(nn.Conv3d(n_feats*self.n_module, n_feats, kernel_size=(1,1,1), stride=1, bias=False))
        
        threefusion = [wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)),
                       wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))]
        self.threefusion = nn.Sequential(*threefusion)

        self.reduceD_DFF = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1, bias=False))  
        self.conv_DFF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1, bias=False)) 
        self.reduceD_FCF = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=(1,1), stride=1, bias=False))  
        self.conv_FCF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1, bias=False))    
    
    def forward(self, x, y, localFeats, i):
        x = x.unsqueeze(1)     
        x = self.ThreeHead(x)    
        skip_x = x         

        y = y.unsqueeze(1)
        y = self.TwoHead(y)
        skip_y = y

        channelX = []
        channelY = []        

        for j in range(self.n_module):        
            x = self.threeCNN[j](x)    
            x = torch.add(skip_x, x)          
            channelX.append(self.gamma_X[j]*x)

            y = self.twoCNN[j](y)           
            y = torch.cat([y, x[:,:,0,:,:], x[:,:,1,:,:], x[:,:,2,:,:]],1)
            y = self.reduceD[j](y)      
            y = torch.add(skip_y, y)         
            channelY.append(self.gamma_Y[j]*y) 
                              
        x = torch.cat(channelX, 1)
        x = self.reduceD_X(x)
        x = self.threefusion(x)
      	                
        y = torch.cat(channelY, 1)        
        y = self.reduceD_Y(y) 
        y = self.twofusion(y)        
     
        y = torch.cat([self.gamma_DFF[0]*x[:,:,0,:,:], self.gamma_DFF[1]*x[:,:,1,:,:], self.gamma_DFF[2]*x[:,:,2,:,:], self.gamma_DFF[3]*y], 1)
       
        y = self.reduceD_DFF(y)  
        y = self.conv_DFF(y)
                       
        if i == 0:
            localFeats = y
        else:
            y = torch.cat([self.gamma_FCF[0]*y, self.gamma_FCF[1]*localFeats], 1) 
            y = self.reduceD_FCF(y)                    
            y = self.conv_FCF(y) 
            localFeats = y  
        y = torch.add(y, skip_y)
        y = self.TwoTail(y) 
        y = y.squeeze(1)   
                
        return y, localFeats  
