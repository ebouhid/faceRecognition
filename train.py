from dataset import TripletFaceDataset
from facerecognitionmodel import FaceRecognitionModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow.pytorch

# Set hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = 8
NUM_EPOCHS = 100
FACE_DIR = 'data/'
MIN_DELTA = 1e-4
PATIENCE = 10
LEARNING_RATE = 5e-4

# Set random seed
seed = 42

# Set transforms
transforms = A.Compose([
    A.Rotate(limit=90, p=0.8),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Resize(299,299),
    A.RandomBrightnessContrast(),
    ToTensorV2(),
])

# Load data
dataset = TripletFaceDataset(root_dir=FACE_DIR, transforms=transforms)

# Get dataloaders
train_loader = DataLoader(dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          drop_last=True)

# Enable autologging
mlflow.pytorch.autolog()
mlflow.log_params({'batch_size': BATCH_SIZE})

# Create model
model = FaceRecognitionModel(lr=LEARNING_RATE, freeze_base=False)

# Log model name as a tag
mlflow.set_tag('encoder_name', model.get_encoder_name())

# Create callbacks
early_stopping = EarlyStopping('train_loss',
                               min_delta=MIN_DELTA,
                               patience=PATIENCE,
                               verbose=True)

checkpoint_callback = ModelCheckpoint(monitor='train_loss', save_top_k=1, mode='min', verbose=True)

# Train model
trainer = pl.Trainer(max_epochs=NUM_EPOCHS, callbacks=[early_stopping, checkpoint_callback])
trainer.fit(model, train_loader)