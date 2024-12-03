from typing import Dict, List, Any, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
import datasets
from transformers import PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig

from prompt import EvalTemplate
from utils import (
    INVALID_ANSWER,
    GMS8KEvaluator, 
    GSM8KParser, 
    sample_answers,
    parse_equations,
)
from data import _apply_template
from torch.utils.data import Dataset, DataLoader


class GSM8KDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer

        # parse answer
        self.dataset = self.dataset.map(
            lambda x: GSM8KParser.get_answer_from_gt(x["answer"])
        )

        # format question
        self.dataset = self.dataset.map(
            _apply_template,
            batched=False,
            fn_kwargs={
                "key": "question",
                "tokenizer": self.tokenizer,
                "add_generation_prompt": True,
            },
            # remove_columns=["quesiton"],
        )
        # format answer
        self.dataset = self.dataset.map(
            _apply_template,
            batched=False,
            fn_kwargs={
                "key": "answer",
                "tokenizer": self.tokenizer,
                "add_generation_prompt": False,
            },
            # remove_columns=["answer"],
        )

        # infer max length
        self.dataset = self.dataset.map(
            lambda x: GSM8KParser.get_question_length(x["question"], tokenizer)
        )
        self.dataset = self.dataset.map(
            lambda x: GSM8KParser.get_answer_length(x["answer"], tokenizer)
        )

        self.max_length_question = max(self.dataset["question_length"])
        self.max_length_answer = max(self.dataset["answer_length"])
        self.max_length = self.max_length_question + self.max_length_answer

        print(f"Maximum answer num_tokens: {self.max_length_answer}")
        print(f"Maximum question num_tokens: {self.max_length_question}")
        print(f"Maximum sequence num_tokens: {self.max_length}")

        self.inf_seq_length = 1024
        print(f"Maximum new tokens in generation: {self.inf_seq_length}")

        # Preprocess the dataset
        self.dataset = self.dataset.map(self._preprocess, batched=False)
        self.dataset.set_format(
            type="pt",
            columns=[
                "input_ids",
                "attention_mask",
                "question_input_ids",
                "question_attention_mask",
                "labels",
            ],
            output_all_columns=True,
        )
        print(f"Setup Completed dataset:\n{self.dataset}")

    def _preprocess(self, instance: Dict[str, Any]):

        out = {}

        question, answer = instance["question"], instance["answer"]

        # tokenize formatted question
        question_encodings = self.tokenizer(
            question, return_tensors="pt", truncation=True
        )

        q_diff = self.max_length_question - question_encodings["input_ids"].shape[1]
        out["question_input_ids"] = torch.concat(
            [
                torch.tensor(self.tokenizer.pad_token_id).repeat(q_diff),
                question_encodings["input_ids"].squeeze(0),
            ]
        )
        out["question_attention_mask"] = torch.concat(
            [
                torch.tensor(0).repeat(q_diff),
                question_encodings["attention_mask"].squeeze(0),
            ]
        )

        # tokenizer formatted answer
        answer_endodings = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
        )

        # concatenate the question and the answer input_ids
        seq_ids = torch.cat(
            [
                question_encodings["input_ids"].squeeze(0),
                answer_endodings["input_ids"].squeeze(0),
            ],
            dim=0,
        )

        # left padd manually
        diff = self.max_length - seq_ids.shape[0]
        if diff > 0:
            # left padding with pad_token_id
            padded_seq_ids = torch.cat(
                [torch.tensor([self.tokenizer.pad_token_id]).repeat(diff), seq_ids],
                dim=0,
            )
        elif diff < 0:
            raise ValueError(
                f"Sequence is too long: {seq_ids.shape[0]} > {self.max_length}"
            )
        else:
            padded_seq_ids = seq_ids
        out["input_ids"] = padded_seq_ids

        # pad_token added has no attention
        # cat( pad span | input_ids span | answer span )
        attention_mask = torch.concat(
            [
                torch.tensor([0]).repeat(diff),
                torch.ones(seq_ids.shape[0]),
            ],
            dim=0,
        )
        assert out["input_ids"].shape[0] == attention_mask.shape[0]
        out["attention_mask"] = attention_mask

        # -100 for (padding + input_ids) and answer_ids for the rest
        labels = torch.concat(
            [
                torch.tensor([-100]).repeat(diff),
                torch.tensor([-100]).repeat(question_encodings["input_ids"].shape[1]),
                answer_endodings["input_ids"].squeeze(0),
            ]
        )
        assert labels.shape[0] == out["input_ids"].shape[0]
        out["labels"] = labels
        return out

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class Phi3LightningModule(pl.LightningModule):

    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 10,
        generation_config: Dict[str, Any] = {},
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
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.generation_config = generation_config
        # Save hyperparameters
        self.save_hyperparameters("learning_rate", "weight_decay", "model_name")

        # Load pretrained model
        self.toknizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

        # self.bb_config = BitsAndBytesConfig(
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            # quantization_config = self.bb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Evaluator
        self.evaluator = GMS8KEvaluator()

        # Buffers
        self.train_step_outs = []
        self.val_step_outs = []
        self.test_step_outs = []

    def forward(self, input_ids:torch.Tensor, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, ttention_mask=attention_mask, labels=labels
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
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Log the loss
        self.log("train_loss", outputs.loss, prog_bar=True, on_step=True)

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
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Log the validation loss
        self.log("val_loss", outputs.loss, prog_bar=True)

        return outputs.loss

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

        # todo: make this maj@k
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
            {"test/batch_maj@1": torch.tensor(maj_accs).float().mean()},
            prog_bar=True,
            logger=True,
            on_step=True,
        )

    def on_test_epoch_end(self):
        epoch_mean_maj_accs = torch.tensor(self.test_step_outs).float().mean()
        self.log_dict(
            {"test/epoch_maj@1": epoch_mean_maj_accs},
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.test_step_outs.clear()

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
            weight_decay=self.hparams.weight_decay,
        )

        # Prepare learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
