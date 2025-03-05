from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F

import random
import numpy as np
import torch


from momentfm import MOMENTPipeline
import random
def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class my_get_model():
    def __init__(self, model_name, freeze = False, dropout = 0.5,pretrain = False, model_path = "empty"):
        self.model_name = model_name
        self.dropout = dropout
        self.pretrain = pretrain
        self.model_path = model_path
        self.freeze = freeze

    def get_model(self):
        model = None
        if(self.model_name == "moment"):             model = self.moment_forecasting()
  
        
        if(self.model_path != "empty"):
            model.load_state_dict(torch.load(self.model_path))
            #model.cuda()


        
        #print(model)
        print("**********************************************************************")
        print(f"laoded model : {self.model_name}")
        print("**********************************************************************")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #summary(model, input_size=(16, 100))
        model = model.to(device)
        return model

    def moment_forecasting(self):
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': 60,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': True,
                'freeze_embedder': True, 
                'freeze_head': False, 
            },
        )
        model.init()
        # print(model)
        
        
        # x = torch.randn(1, 1, 512)
        # output = model(x)
        # print(output)
        #exit(0)
        return model


