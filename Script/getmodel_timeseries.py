from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F

import random
import numpy as np
import torch
from torchsummary import summary
from transformers import InformerForPrediction

from transformers import TimeSeriesTransformerConfig
from transformers import InformerConfig
from transformers import InformerModel


from momentfm import MOMENTPipeline
from torchtsmixer import TSMixer
import random
def set_seed(seed):
    # Python 시드 설정
    random.seed(seed)

    # Numpy 시드 설정
    np.random.seed(seed)

    # PyTorch 시드 설정
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN 알고리즘 시드 설정 (선택 사항)
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
        if(self.model_name == "informer"):           model = self.Informer()
        if(self.model_name == "moment"):             model = self.moment_forecasting()
        if(self.model_name == "TSMixer"):            model = self.TSMixer()
        
        
        if(self.model_path != "empty"):
            model.load_state_dict(torch.load(self.model_path))
            model.cuda()


        
        #print(model)
        print("**********************************************************************")
        print(f"laoded model : {self.model_name}")
        print("**********************************************************************")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #ummary(model, input_size=(16, 100))
        model = model.to(device)
        
        return model.cuda()

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
        
        
        #x = torch.randn(1, 1, 512)
        #output = model(x)
        #print(output)
        #exit(0)
        return model

    def Informer(self):
        # config = InformerConfig(
        #         input_size = 1,
        #         prediction_length = 360,
        #         context_length= 600,
        #         num_time_features=5,
        #         lags_sequence=[1,2,3,4,5,6,7,8,9,10,600],
        #         scaling='std'
        #     )
        config = InformerConfig(
                input_size = 1,
                prediction_length = 60,
                context_length= 180,
                num_time_features=5,
                lags_sequence=[1,2,3,4,5,6,7,8,9,10,180],
                scaling='std'
            )
        model = InformerForPrediction(config)
        return model

    def TSMixer(self):
        model = TSMixer(
            sequence_length=512,
            prediction_length=60,
            input_channels=18,
            output_channels=1
        )
        return model
