# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from models.resnet_se import ResNetSE
class resnetse(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = ResNetSE(112, layers=[3, 4, 6, 3], num_filters=[32, 64, 128, 256], embd_dim=512,
                 pooling_type="ASP")
        
        

    def forward(self, x):
        """
        :param x: input mini-batch (bs, samp)
        """
        x=x.squeeze(1).to("cuda")
        #print(x.shape)
        x=self.model(x)
        
        


        return x



def MainModel(**kwargs):

    model = resnetse(
       
    )
    return model
    
    
    
    
   
          
 
      
   
