import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn import TripletMarginLoss
from pytorch_metric_learning.losses import TripletMarginLoss as MetricLearningTripletLoss
from pytorch_metric_learning.miners import BatchHardMiner, MultiSimilarityMiner
import math


class FaceRecognitionModel(pl.LightningModule):
    def __init__(self, lr=1e-4, encoder='efficientnet_b3', embedding_dim=512, input_dim=224, miner_type='batch_hard', margin=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.encoder_name = encoder
        self.embedding_dim = embedding_dim
        self.miner_type = miner_type
        self.margin = margin
        self.input_dim = input_dim
        
        # Create the base model and extract features properly
        base_model = timm.create_model(self.encoder_name, pretrained=True, num_classes=0)
        self.encoder = base_model
        
        # Get the number of features from the encoder
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.input_dim, self.input_dim)
            dummy_output = self.encoder(dummy_input)
            num_features = dummy_output.shape[1]
        
        # Enhanced projection head
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim)
        )
        
        self.distance = torch.nn.PairwiseDistance(p=2)
        self.criterion = MetricLearningTripletLoss(margin=self.margin)
        # Miner selection
        if miner_type == 'batch_hard':
            self.miner = BatchHardMiner()
        elif miner_type == 'multi_similarity':
            self.miner = MultiSimilarityMiner()
        else:
            raise ValueError(f"Unknown miner_type: {miner_type}")
        self.lr = lr

    def forward(self, x):
        # Extract features using the encoder
        features = self.encoder(x)
        
        # Apply projection head
        embedding = self.projection_head(features)
        
        # L2 normalize the embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        # Concatenate all samples and create labels
        embeddings = torch.cat([self(anchor), self(positive), self(negative)], dim=0)
        batch_size = anchor.size(0)
        # Labels: [0, 1, ..., batch_size-1] for anchor, same for positive, same for negative
        labels = torch.arange(batch_size, device=self.device)
        labels = labels.repeat(3)  # [A0, A1, ..., P0, P1, ..., N0, N1, ...]
        # Mine hard triplets
        hard_triplets = self.miner(embeddings, labels)
        loss = self.criterion(embeddings, labels, hard_triplets)
        # For logging, compute distances for anchors/positives/negatives
        emb_a = embeddings[:batch_size]
        emb_p = embeddings[batch_size:2*batch_size]
        emb_n = embeddings[2*batch_size:]
        pos_distance = self.distance(emb_a, emb_p).mean()
        neg_distance = self.distance(emb_a, emb_n).mean()
        margin = self.criterion.margin
        correct_triplets = (pos_distance + margin < neg_distance).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_pos_dist', pos_distance, on_step=True, prog_bar=False)
        self.log('train_neg_dist', neg_distance, on_step=True, prog_bar=False)
        self.log('train_triplet_acc', correct_triplets, on_step=False, on_epoch=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        embeddings = torch.cat([self(anchor), self(positive), self(negative)], dim=0)
        batch_size = anchor.size(0)
        labels = torch.arange(batch_size, device=self.device)
        labels = labels.repeat(3)
        hard_triplets = self.miner(embeddings, labels)
        loss = self.criterion(embeddings, labels, hard_triplets)
        # For logging, compute distances for anchors/positives/negatives
        emb_a = embeddings[:batch_size]
        emb_p = embeddings[batch_size:2*batch_size]
        emb_n = embeddings[2*batch_size:]
        pos_distance = self.distance(emb_a, emb_p).mean()
        neg_distance = self.distance(emb_a, emb_n).mean()
        margin = self.criterion.margin
        correct_triplets = (pos_distance + margin < neg_distance).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pos_dist', pos_distance, on_step=False, on_epoch=True)
        self.log('val_neg_dist', neg_distance, on_step=False, on_epoch=True)
        self.log('val_triplet_acc', correct_triplets, on_step=False, on_epoch=True, prog_bar=True)
        return loss
        

    def configure_optimizers(self):
        # Use AdamW for better regularization
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Use Cosine Annealing with Warm Restarts for better convergence
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6),
            'interval': 'epoch',
            'frequency': 1,
            'name': 'cosine_annealing'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_embedding(self, x):
        return self.encoder(x)
    
    def get_encoder_name(self):
        return self.encoder_name

class ResNetFaceModel(pl.LightningModule):
    def __init__(self, lr=1e-4, embedding_dim=512):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_dim = embedding_dim
        
        # Use ResNet50 as backbone - proven for face recognition
        self.encoder_name = 'resnet50'
        backbone = timm.create_model(self.encoder_name, pretrained=True, num_classes=0)
        self.encoder = backbone
        
        # Get number of features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 112, 112)
            num_features = self.encoder(dummy_input).shape[1]
        
        # Face-specific projection head
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),  # PReLU works better for face recognition
            nn.Dropout(0.4),
            
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=False)  # No learnable parameters in final BN
        )
        
        self.criterion = TripletMarginLoss(margin=0.3, p=2)  # Smaller margin
        self.distance = torch.nn.PairwiseDistance(p=2)
        self.lr = lr

    def forward(self, x):
        features = self.encoder(x)
        embedding = self.projection_head(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        emb_a, emb_p, emb_n = self(anchor), self(positive), self(negative)
        loss = self.criterion(emb_a, emb_p, emb_n)
        
        # Additional metrics
        pos_dist = self.distance(emb_a, emb_p).mean()
        neg_dist = self.distance(emb_a, emb_n).mean()
        acc = (pos_dist + self.criterion.margin < neg_dist).float().mean()
        
        self.log_dict({
            'train_loss': loss,
            'train_pos_dist': pos_dist,
            'train_neg_dist': neg_dist,
            'train_acc': acc
        }, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        emb_a, emb_p, emb_n = self(anchor), self(positive), self(negative)
        loss = self.criterion(emb_a, emb_p, emb_n)
        
        pos_dist = self.distance(emb_a, emb_p).mean()
        neg_dist = self.distance(emb_a, emb_n).mean()
        acc = (pos_dist + self.criterion.margin < neg_dist).float().mean()
        
        self.log_dict({
            'val_loss': loss,
            'val_pos_dist': pos_dist,
            'val_neg_dist': neg_dist,
            'val_acc': acc
        }, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-4)
        scheduler = {
        'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3),
        'monitor': 'val_loss',
        'interval': 'epoch',
        'frequency': 1,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def get_encoder_name(self):
        return self.encoder_name
