import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.SurgicalActionNet import SurgicalVerbRecognition
from utils.verbdataloader import VerbDataModule
import os

def main():
    # Basic setup
    base_dir = "/data/Bartscht/Verbs"
    batch_size = 16
    num_workers = 8
    
    # Initialize data and model
    print("Initializing DataModule...")
    datamodule = VerbDataModule(base_dir, batch_size, num_workers)
    
    print("Initializing Model...")
    model = SurgicalVerbRecognition(
        num_classes=10,
        learning_rate=1e-5,
        backbone_learning_rate=1e-6,
        dropout=0.5
    )
    
    # Initialize WandB logger first to get the run ID
    wandb_logger = WandbLogger(project='verb-recognition')
    run_id = wandb_logger.experiment.id
    run_name = wandb_logger.experiment.name
    
    # Create a unique directory for this run's checkpoints
    checkpoint_dir = os.path.join('checkpoints', f'{run_name}-{run_id}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Updated checkpoint callback with run information
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'verb-model-{run_name}-{run_id}' + '-epoch={epoch:02d}-val_loss={val/loss:.3f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    # Simple trainer setup
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        log_every_n_steps=10
    )
    
    # Start training
    try:
        print(f"Starting training... Run ID: {run_id}, Run Name: {run_name}")
        trainer.fit(model, datamodule)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()