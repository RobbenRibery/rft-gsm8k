import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import wandb

class GSM8KDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        """
        Custom dataset for GSM8K
        
        Args:
            dataset: Hugging Face dataset
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Preprocess the dataset
        self.processed_data = self._preprocess()
    
    def _preprocess(self):
        """
        Preprocess the dataset by tokenizing
        
        Returns:
            List of tokenized examples
        """
        processed = []
        for item in self.dataset:
            # Combine question and answer
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            
            # Tokenize
            encodings = self.tokenizer(
                text, 
                truncation=True, 
                padding='max_length', 
                max_length=self.max_length, 
                return_tensors='pt'
            )
            
            processed.append({
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()
            })
        
        return processed
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

class Phi3LightningModule(pl.LightningModule):
    def __init__(
            self, 
            model_name:str='microsoft/Phi-3-mini-4k-instruct',
            learning_rate:float=1e-5,
            weight_decay:float=0.01,
        ):
                 
        """
        Lightning module for Phi-3 training
        
        Args:
            model_name: Pretrained model name
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps for learning rate scheduler
        """
        super().__init__()

        self.lr = learning_rate 
        # Save hyperparameters
        self.save_hyperparameters(
            'learning_rate',
            'weight_decay',
            'model_name'
        )
        
        # Load pretrained model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
        
        Returns:
            Model outputs
        """
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Current batch of data
            batch_idx: Batch index
        
        Returns:
            Computed loss
        """
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels']
        )
        
        # Log the loss
        self.log('train_loss', outputs.loss, prog_bar=True, on_step=True)
        
        return outputs.loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch: Current batch of data
            batch_idx: Batch index
        
        Returns:
            Computed loss
        """
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels']
        )
        
        # Log the validation loss
        self.log('val_loss', outputs.loss, prog_bar=True)
        
        return outputs.loss
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler
        
        Returns:
            Optimizer and learning rate scheduler
        """
        # Prepare optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        
        # Prepare learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset = load_dataset('gsm8k', 'main')['train']
    val_dataset = load_dataset('gsm8k', 'main')['test']
    
    # Create custom datasets
    train_data = GSM8KDataset(train_dataset, tokenizer)
    val_data = GSM8KDataset(val_dataset, tokenizer)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_data, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=4, 
        num_workers=4
    )
    
    wandb_logger = WandbLogger(
        project='phi3-gsm8k-training',
        log_model='all'
    )
    
    # Model Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='phi3-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    # Early Stopping Callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )
    
    # Initialize Lightning Module
    model = Phi3LightningModule()
    
    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',  # Mixed precision training
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm'
    )
    
    # Train the model
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()