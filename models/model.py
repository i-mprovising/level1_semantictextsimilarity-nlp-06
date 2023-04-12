import torch
import torchmetrics
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self, CFG, plm):
        self.save_hyperparameters()
        
        self.model_name = CFG.train.model_name
        self.lr = CFG.train.LR
        
        # 사용할 모델을 호출
        self.plm = plm
        
        # Loss 계산을 위해 사용될 L1Loss를 호출
        self.loss_func = torch.nn.L1Loss()
    
    def forward(self, x):
        x = self.plm(x)['logits']
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log('train_loss >>', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log('val_loss >>', loss)
        self.log('val_pearson >>', torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        
        return logits.squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        return optimizer