from dataset import TripletFaceDataset
from facerecognitionmodel import FaceRecognitionModel, ResNetFaceModel
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import timm
import random

torch.set_float32_matmul_precision('medium')

# Set hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 16
NUM_EPOCHS = 100
FACE_DIR = '/home/ebneto/Downloads/digiface_1m/'
# FACE_DIR = 'lfw_cropped/'
MIN_DELTA = 1e-3
ENCODER='resnet50'
PATIENCE = 7
VALIDATION_SPLIT = 0.2
LR = 1e-4
EMBEDDING_DIM = 512
IMAGE_DIM = 224

# # Set random seed
seed = 42
pl.seed_everything(seed)  # Enable full reproducibility

temp_model = timm.create_model(ENCODER, pretrained=True)
DATA_MEAN = temp_model.default_cfg['mean']
DATA_STD = temp_model.default_cfg['std']
del temp_model

# Set transforms
train_transforms = A.Compose([
    A.RandomResizedCrop((IMAGE_DIM, IMAGE_DIM), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.7),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.4),
    A.RGBShift(r_shift_limit=0.3, g_shift_limit=0.3, b_shift_limit=0.3, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    # A.CoarseDropout(max_holes=2, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, fill_value=0, p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Resize(IMAGE_DIM, IMAGE_DIM),
    A.Normalize(mean=DATA_MEAN, std=DATA_STD),
    ToTensorV2()
])
val_transforms = A.Compose([
    A.Resize(IMAGE_DIM, IMAGE_DIM),
    A.Normalize(mean=DATA_MEAN, std=DATA_STD),
    ToTensorV2()
])

# Create separate dataset instances for train and validation
train_dataset = TripletFaceDataset(root_dir=FACE_DIR, transforms=train_transforms, debug=False)
val_dataset = TripletFaceDataset(root_dir=FACE_DIR, transforms=val_transforms, debug=False)

# Set deterministic mode for validation
val_dataset.set_deterministic_mode(True)

# For proper train/val split, we need to split at the celebrity level, not image level
# This prevents data leakage where same celebrity appears in both train and val
celebrities = list(train_dataset.celebrity_to_paths_map.keys())
random.seed(seed)  # For reproducible celebrity split
random.shuffle(celebrities)

val_celebrities_count = int(VALIDATION_SPLIT * len(celebrities))
val_celebrities = set(celebrities[:val_celebrities_count])
train_celebrities = set(celebrities[val_celebrities_count:])

# Filter datasets to only include their respective celebrities
train_dataset.filter_by_celebrities(train_celebrities)
val_dataset.filter_by_celebrities(val_celebrities)

# Debug: Print some statistics to verify the split
print(f"Total celebrities: {len(celebrities)}")
print(f"Train celebrities: {len(train_celebrities)}")
print(f"Val celebrities: {len(val_celebrities)}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")

# Get dataloaders
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          drop_last=True)

val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS)

# Create model
# model = FaceRecognitionModel(lr=LR, encoder=ENCODER, embedding_dim=EMBEDDING_DIM, miner_type='multi_similarity')
model = ResNetFaceModel(lr=LR, embedding_dim=EMBEDDING_DIM)

# Create callbacks
early_stopping = EarlyStopping('val_loss',
                               min_delta=MIN_DELTA,
                               patience=PATIENCE,
                               verbose=True,
                               mode='min')

checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', verbose=True, save_last=True)

# Train model
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS, 
    callbacks=[early_stopping, checkpoint_callback], 
    log_every_n_steps=20,
    gradient_clip_val=1.0,  # Add gradient clipping for stability
    deterministic=True  # For reproducibility
)
trainer.fit(model, train_loader, val_loader)
