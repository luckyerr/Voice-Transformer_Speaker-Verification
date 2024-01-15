# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from models.campplus import CAMPPlus
class campplus(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = CAMPPlus(
                 112,
                 embd_dim=512,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True)
        
        

    def forward(self, x):
        """
        :param x: input mini-batch (bs, samp)
        """
        x=x.squeeze(1).to("cuda")
        x=self.model(x)
        #print(x.shape)
        


        return x



def MainModel(**kwargs):

    model = campplus(
       
    )
    return model
    
    
    
    
   
          
 
      
   
