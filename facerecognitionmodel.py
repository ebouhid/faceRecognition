import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm

class Encoder(torch.nn.Module):
    def __init__(self, base_model, out_dim, freeze_inception=True):
        super().__init()

        # Load the TIMM base model
        self.model = timm.create_model(base_model, pretrained=True, num_classes=1000)

        # Freeze the base model
        if freeze_inception:
            for param in self.model.parameters():
                param.requires_grad = False

        # Add a custom trainable CNN head to reduce features to out_dim
        self.cnn_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(self.model.num_features, out_dim)
        )
        for param in self.cnn_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Forward pass through the TIMM model
        features = self.model(x)

        # Forward pass through the custom CNN head
        embedding = self.cnn_head(features)

        return embedding

class FaceRecognitionModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.encoder_name = 'xception65.tf_in1k'
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