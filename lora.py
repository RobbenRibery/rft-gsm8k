from typing import Dict, List, Any, Tuple
import os
import torch
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch import seed_everything

import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel, 
    PreTrainedTokenizer, 
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)

from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model,
    peft_model,
)

import bitsandbytes as bnb

from dataclasses import dataclass

from data import GSM8KDataset
from utils import (
    INVALID_ANSWER,
    GMS8KEvaluator, 
    GSM8KParser, 
    sample_answers,
)
from torch.utils.data import DataLoader
import tyro

@dataclass
class TrainingConfig:
    learning_rate: float = 5e-03
    weight_decay: float = 0.01
    batch_size:int = 64
    max_n_epochs: int = 1
    gradient_accum_steps: int = 8
    val_interval:float = 0.1


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    return_dict_in_generate: bool = True

@dataclass
class LoraConfig:
    task_type: TaskType = TaskType.CAUSAL_LM
    r: int = 32
    lora_alpha: int = 32

    #https://arxiv.org/pdf/2312.03732 sugges that rslora would 
    # yield better distinction when ranks go up
    use_rslora:bool = True 
    
    lora_dropout: float = 0.05
    bias: str = "none"
    inference_mode: bool = False

class LoraLit(pl.LightningModule):

    def __init__(
        self,
        model_name: str,
        train_dataset: GSM8KDataset,
        training_config: TrainingConfig,
        generation_config: GenerationConfig,
        lora_config: LoraConfig,
        lora_target_modules:List[str] = [
            "gate_proj", 
            "down_proj", 
            "up_proj"
        ],
        model_output_dir: str = "model_output",
        test_base_model:bool = False,
    ):
        super().__init__()

        self.model_name = model_name

        self.train_config = training_config
        self.lora_config = lora_config
        self.generation_config = generation_config
        self.lora_target_modules = lora_target_modules

        self.train_dataset = train_dataset

        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

        self.test_base_model = test_base_model
        
        # Save hyperparameters
        self.save_hyperparameters("train_config", "lora_config", "generation_config")

        # Load model and tokenizer 
        self.toknizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

        # quantization config 
        self.bb_config = BitsAndBytesConfig(
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config = self.bb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Lora config
        lora_config = LoraConfig(
            target_modules = self.lora_target_modules,
            **self.lora_config().__dict__
        )

        # Quantization
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True,
        )
        self.model:peft_model.PeftModel = get_peft_model(self.model, lora_config)

        # Evaluator
        self.evaluator = GMS8KEvaluator()

        # Buffers
        self.train_step_outs = []
        self.val_step_outs = []
        self.test_step_outs = []

    def forward(
        self, 
        input_ids:torch.Tensor, 
        attention_mask:torch.Tensor=None, 
        labels:torch.Tensor=None
        ):
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
    
    ## -- check pointing -- ## 
    def on_load_checkpoint(self, checkpoint):
        checkpoint.update({
            'model_name': self.model_name,
            'peft_state_dict': self.model.state_dict(),
            'model_output_dir': str(self.model_output_dir)
        })
        
        # Save adapter weights alongside checkpoint
        adapter_path = Path(self.trainer.checkpoint_callback.dirpath) / f"adapter_epoch_{self.current_epoch}"
        self.model.save_pretrained(adapter_path)

    def on_load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint['peft_state_dict'])
        self.model_output_dir = Path(checkpoint['model_output_dir'])

    ## -- trian steps -- ## 
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Log the loss
        self.log("train/train_loss_step", outputs.loss, prog_bar=True, on_step=True, logger=True)
        return {"train_loss": outputs.loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.log("train/train_loss_epoch", avg_loss, prog_bar=True, on_epoch=True, logger=True)

    def on_train_end(self):
        
        # define paths
        final_adapter_path = self.model_output_dir / "final_adapter"
        final_model_path = self.model_output_dir / "final_model"

        # save the adapter
        self.model.save_pretrained(final_adapter_path)

        # save configs 
        config = {
            "model_name": self.model_name,
            "adapter_path": str(final_adapter_path),
            "merged_path": str(final_model_path)
        }
        torch.save(config, self.save_path / "model_config.pt")
        
        # merge and save
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(self.save_path / final_model_path)
        print(f"Merged model saved to {final_model_path}")

    ## -- validation steps -- ##
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        # Log the validation loss
        self.log("train/val_loss_step", outputs.loss, prog_bar=True, on_step=True, logger=True)
        return {"val_loss": outputs.loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("train/val_loss_epoch", avg_loss, prog_bar=True, on_epoch=True, logger=True)


    ## -- test steps -- ##
    def on_test_start(self):

        if self.test_base_model:
            self.model.eval()
            return 

        # load the config first 
        config_path = self.model_output_dir / "model_config.pt"
        config = torch.load(config_path)

        # load the trained model 
        self.model = AutoModelForCausalLM.from_pretrained(
            config["merged_path"],
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.model.eval()


    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:

        assert self.generation_config["num_return_sequences"] == 1
        input_ids = batch["question_input_ids"]
        attention_mask = batch["question_attention_mask"]
        refs = batch["answer_str_digit"]

        outs: List[str] = sample_answers(
            self.toknizer,
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **self.generation_config,
        )

        # calculate maj@1
        preds = [
            GSM8KParser.get_answer_from_pred(out)["answer_str_digit"] \
            for out in outs
        ]
        maj_accs = [
            self.evaluator.get_maj_at_k(pred, ref) \
            for pred, ref in zip(preds, refs)
        ]
        self.test_step_outs.extend(maj_accs)
        self.log_dict(
            {"test/maj@1_step": torch.tensor(maj_accs).float().mean()},
            prog_bar=True,
            logger=True,
            on_step=True,
        )

    def on_test_epoch_end(self):
        epoch_mean_maj_accs = torch.tensor(self.test_step_outs).float().mean()
        self.log_dict(
            {"test/maj@1_epoch": epoch_mean_maj_accs},
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.test_step_outs.clear()

    ## -- Optimization -- ## 
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler

        Returns:
            Optimizer and learning rate scheduler
        """
        # Prepare optimizer we use 8bit to save memory
        optimizer = bnb.optim.Adam8bit(
            self.parameters(),
            lr=self.train_config.learning_rate,
        )
        total_training_steps = (
            len(self.train_dataset)//self.train_config.batch_size
        )*self.train_config.max_n_epochs
    
        #  Prepare learning rate scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=total_training_steps,
            num_warmup_steps=int(total_training_steps*0.02)
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    
    # Cli 
    @dataclass
    class Args: 
        rft_data_path: str = "gsm8k_synthetic_data_74instances_5samples"
        """Path to the RFT dataset, sampled from the base model"""

        test_base_model:bool = False
        """Only test the base model"""

        run_name: str = None 
        """Name of the run to log to wandb"""

        seed:int = 42
        """Mannual Seed for the run"""

    # parse
    args = tyro.cli(Args)

    # seeding
    seed_everything(args.seed)

    # load the rft data if provided
    if args.rft_data_path:
        rft_data = datasets.load_from_disk(args.rft_data_path)
        rft_data = rft_data.rename_column("favored_solutions","answer")
        args.run_name = "qlora_rft_"+args.rft_data_path.split("_")[-2]
    else:
        args.run_name = "qlora_sft"

    # setup
    MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Get Train & Val DataLoader
    train_dataset = datasets.load_dataset('gsm8k', 'main')['train']
    TrainData = GSM8KDataset(train_dataset, tokenizer)
    TrainLoader = DataLoader(
        train_dataset,
        batch_size=TrainingConfig.batch_size,
        shuffle=True,
        num_workers= os.cpu_count(),
    )

    val_dataset = datasets.load_dataset('gsm8k', 'main')['test']
    ValData = GSM8KDataset(val_dataset, tokenizer)
    ValLoader = DataLoader(
        val_dataset,
        batch_size=TrainingConfig.batch_size,
        shuffle=False,
        num_workers= os.cpu_count(),
    )

    LoraModel = LoraLit(
        model_name=MODEL_NAME,
        model_output_dir="./{args.run_name}",
        train_dataset=TrainData,
        training_config=TrainingConfig,
        generation_config=GenerationConfig,
        lora_config=LoraConfig,
        lora_target_modules=[
            "gate_proj", 
            "down_proj", 
            "up_proj"
        ],
        test_base_model=args.test_base_model,
    )

    # logg to wandb
    wandb_logger = WandbLogger(
        project="phi3-gsm8k-rft", 
        log_model="all",
        name=args.run_name,
    )

    # callbacks
    pbar = RichProgressBar()


    # callbacks
    ckpts = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        save_last=True,
        save_top_k=1,
        enable_version_counter=True,
    )

    early_stop = EarlyStopping(
        monitor="train/val_loss_step",
        mode="min",
        patience=3,
    )

    # trainer
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=[pbar],
        max_epochs=TrainingConfig.max_n_epochs,
        min_epochs=1,
        accumulate_grad_batches= TrainingConfig.gradient_accum_steps,
        precision="bf16",
        val_check_interval= TrainingConfig.val_interval,
        log_every_n_steps=1,
        enable_progress_bar=True,
        deterministic=True,
    )

    if args.test_base_model:
        trainer.test(LoraModel, dataloaders=ValLoader)

    else:
        trainer.fit(
            LoraModel,
            train_dataloader=TrainLoader,
            val_dataloaders=ValLoader
        )
        best_lora = trainer.test(ckpt_path="best")