# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from models.ecapa_tdnn import EcapaTdnn
class ecapatdnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = EcapaTdnn(input_size=112, channels=512, embd_dim=512, pooling_type="ASP")
        
        

    def forward(self, x):
        """
        :param x: input mini-batch (bs, samp)
        """
        x=x.squeeze(1).to("cuda")
        x=self.model(x)
        #print(x.shape)
        


        return x



def MainModel(**kwargs):

    model = ecapatdnn(
       
    )
    return model
    
    
    
    
   
          
 
      
   
