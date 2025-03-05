from Univariate_Dataset import MY_Coin_Dataloader
from MultiVariate_Dataloader import MV_Coin_Dataloader
from getmodel_timeseries import my_get_model
import os
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import random
import time
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
torch.backends.cudnn.benchmark = True

class moment_Trainer:
    def __init__(self
                 ,Dataloader
                 ,data_dir
                 ,model_name
                 ,epochs
                 ,saved_model_path
                 ,save_test_result_path
                 ) -> None:
        
        Getmodel_obj = my_get_model(model_name)
        self.model = Getmodel_obj.get_model()
        self.saved_model_path = saved_model_path
        self.save_test_result_path = save_test_result_path 
        self.milstone = 0
        self.epochs = epochs
        if(len(os.listdir(self.saved_model_path))!=0):
            last_model_names = os.listdir(self.saved_model_path)
            last_model_name = max(last_model_names, key=lambda x: int(x.split('.')[0]))
            last_model_path = os.path.join(self.saved_model_path, last_model_name)
            self.model.load_state_dict(torch.load(last_model_path))          
            self.milstone = int(last_model_name.split('.')[0])

        self.data_dir = data_dir
        self.Dataloader = Dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           

    def train(self,optimizer,criterion):
        optim = optimizer(self.model.parameters(), lr=1e-4)
        self.model.to(self.device)
        train_loader = self.Dataloader(self.data_dir,mode = "train",batch_size = 32)
        train_loader_length = len(train_loader)
        torch.cuda.empty_cache()
        batch_train_loss = []
        
        epoch_train_loss = []
        epoch_valid_loss = []
        for epoch in range(self.milstone,self.epochs):
            self.model.train(True)
            # self.test(epoch)
            # exit(0)
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{self.epochs} - train process')
            for idx, (batch) in progress_bar:
                
      
                
                past_values,future_values, past_observed_mask= batch['past_values'],batch['future_values'],batch['past_observed_mask']
                past_values = past_values.to(self.device, non_blocking=True)
                future_values = future_values.to(self.device, non_blocking=True)
                past_observed_mask = past_observed_mask.to(self.device, non_blocking=True)
                
                
                with torch.cuda.stream(torch.cuda.Stream()):
                    start_time = time.time()
                    #print("시작")      
                    outputs = self.model(x_enc = past_values.unsqueeze(1), input_mask = past_observed_mask)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    #print(f"Loss calculation took {elapsed_time:.6f} seconds to complete.")              
                    output_close = outputs.forecast[:,0,:]
    
                    loss = criterion(output_close, future_values)
                

                
                #atch_train_loss.append(loss.item())
                loss.backward()
                optim.step()
                optim.zero_grad()
                progress_bar.set_postfix(loss=loss.item())
                
                

                
                
            train_loss = sum(batch_train_loss)/train_loader_length
            epoch_train_loss.append(train_loss)   
        
            self.model.train(False)
            with torch.no_grad():  
                torch.save(self.model.state_dict(), f'{self.saved_model_path}/{epoch}.pt')
                self.test(epoch)
           

        self.train_matrix = {
            "epoch_train_loss" : epoch_train_loss,
            "epoch_valid_loss" : epoch_valid_loss,
        }  
              
    def valid(self,epoch):
        torch.cuda.empty_cache()
        valid_loader = self.Dataloader(data_dir,mode = "valid",batch_size=1024)
        valid_loader_length = len(valid_loader)
        total_valid_loss = []
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Epoch {epoch+1}/{self.epochs} - validation process', leave=False)
        for idx, (batch) in progress_bar:
            loss = self.calc_batch(self.model, batch,"Train")
            total_valid_loss.append(loss)
            progress_bar.set_postfix(loss=loss.item())
        
        return sum(total_valid_loss)/valid_loader_length
    
    def test(self,epoch):
        test_loader = self.Dataloader(data_dir,mode = "test",batch_size=1)
        test_loader_length = len(test_loader)
        total_test_loss = []
        preds = []
        labels = []
        for idx, (batch) in tqdm(enumerate(test_loader),f'Epoch {epoch+1}/{self.epochs} - Test process'):
            #predict_sequence = self.calc_batch(self.model, batch,criterion,mode = "Inference")
            past_values,future_values, past_observed_mask= batch['past_values'],batch['future_values'],batch['past_observed_mask']
            past_values = past_values.to(self.device, non_blocking=True)
            future_values = future_values.to(self.device, non_blocking=True)
            past_observed_mask = past_observed_mask.to(self.device, non_blocking=True)

            
            with torch.cuda.stream(torch.cuda.Stream()):
                start_time = time.time()
                print("시작")      
                outputs = self.model(x_enc = past_values.unsqueeze(1), input_mask = past_observed_mask)
                #outputs = self.model(x_enc = past_values,input_mask = past_observed_mask)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Loss calculation took {elapsed_time:.6f} seconds to complete.")              
                output_close = outputs.forecast[:,0,:]

            
            
            pred_close = output_close.squeeze(0).squeeze(0).detach().cpu().numpy()
            label_close = future_values.squeeze(0).squeeze(0).detach().cpu().numpy()

            preds.append(pred_close)
            labels.append(label_close)
            break

        self.plot_fig(
            np.concatenate(preds, axis=0),
            np.concatenate(labels, axis=0),
            #np.concatenate(times,axis=0),
            epoch,
            self.save_test_result_path
        )

        return sum(total_test_loss)/test_loader_length
    
    def plot_fig(self,pred_close,label_close,epoch,save_dir):
        # start_time = time_feature[0]
        # start_time = datetime(int(start_time[0]), int(start_time[1]), int(start_time[2]), int(start_time[3]), int(start_time[4]))
        # end_time = time_feature[-1]
        # end_time = datetime(int(end_time[0]), int(end_time[1]), int(end_time[2]), int(end_time[3]), int(end_time[4]))
        # # datetime 객체를 원하는 형식의 문자열로 변환
        # str_start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
        # str_end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')

        plt.figure()  # 새로운 그림 생성
        plt.figure(figsize=(60,7))
        plt.plot(pred_close, color='red', label='pred',linewidth =1.4,linestyle='--')
        plt.plot(label_close, color='blue', label='truth',linewidth =1.4, linestyle='--')
        #plt.title(f'{epoch} -- {str_start_time} ~ {str_end_time} predict',fontsize=20)
        plt.title(f'{epoch} -- first predict',fontsize=20)
        plt.xlabel("time", fontsize=20)
        plt.ylabel("prise", fontsize=20)
        plt.legend()
        plt.savefig(f"{save_dir}/{epoch+1}.png")
        plt.close()
        
    def calc_batch(self, model, batch,criterion, mode = "Train"):
        torch.cuda.synchronize()
     
   


 

        with torch.cuda.stream(torch.cuda.Stream()):
            outputs = model(x_enc = past_values,input_mask = past_observed_mask)
            output_close = outputs.forecast[:,0,:]
            loss = criterion(output_close, future_values)



        if mode == "Train":

            return loss
        elif mode == "Inference":
            return output_close
        else:
            raise ValueError("invalid Calculation mode name")
        
data_dir = "/Capston_Design/DataParser/PreProcessing_data.csv"
save_model_path = "/Capston_Design/DataParser/result/moment_512_60_model"
test_result_path = "/Capston_Design/DataParser/result/moment_512_60_model_test_plot"

criterion = nn.L1Loss()
trainer = moment_Trainer(
    MY_Coin_Dataloader,
    data_dir,
    "moment",
    1000,
    save_model_path,
    test_result_path
)
trainer.train( 
    torch.optim.AdamW,
    criterion
)


