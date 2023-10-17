import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm

class Encoder(torch.nn.Module):
    def __init__(self, model_name, out_dim):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=out_dim)

    def forward(self, x):
        return self.model(x)

class FaceRecognitionModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.encoder_name = 'resnet34'
        self.encoder = Encoder(self.encoder_name, 128)
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)
        self.lr = lr
    

    def forward(self, anchor, positive, negative):
        embedding_anchor = self.encoder(anchor)
        embedding_positive = self.encoder(positive)
        embedding_negative = self.encoder(negative)

        return embedding_anchor, embedding_positive, embedding_negative
    
    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor, positive, negative = self.forward(anchor, positive, negative)
        loss = self.criterion(anchor, positive, negative)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {'optimizer': optimizer, 'lr_scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)}

    def get_embedding(self, x):
        return self.encoder(x)
    
    def get_encoder_name(self):
        return self.encoder_name