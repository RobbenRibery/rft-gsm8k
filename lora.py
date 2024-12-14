import os
from typing import List, Dict, Any 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.nn import CrossEntropyLoss
from lightning.pytorch import seed_everything

import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel, 
    PreTrainedTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorMixin

from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)

from trl import SFTTrainer
from dataclasses import dataclass

from data import GSM8KDataset, filter_dataset
from utils import GSM8KParser, sample_answers
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd  

import tyro
os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclass
class TrainingConfig:

    learning_rate: float = 1e-03
    """Learning rate during training"""

    batch_size:int = 32
    """Per-device batch size"""

    max_n_epochs: int = 1
    """"Maximum number of epochs"""

    gradient_accum_steps: int = 1
    """Number of gradient accumulation steps"""

    val_interval:float = 0.05
    """Percentage of steps trigeering validation"""

@dataclass
class GenerationConfig:

    max_new_tokens: int = 1024
    """"
    Maxium number of new tokens to generate
    (This will override the max_if_length
    """
    temperature: float = 0.2
    """Sampling Temperature during inference"""

    return_dict_in_generate: bool = True
    """"Return inference results as a dictionary or not"""

    num_return_sequences:int = 1
    """Number of completions to generate"""

    do_sample: bool = True
    """Use sampling during inference"""

@dataclass
class LoraLayersConfig:
    
    task_type: TaskType = TaskType.CAUSAL_LM
    """Task type for LoRA training"""
    
    r: int = 16
    """"Rank of the LoRa matrices"""
    
    lora_alpha: int = int(r*0.8)
    """"Scaling factor"""
    
    use_rslora:bool = True 
    """"
    #https://arxiv.org/pdf/2312.03732 sugges that rslora would 
    # yield better distinction when ranks go up
    """

    lora_dropout: float = 0.1
    """"Dropout probability for LoRA layers"""

    bias: str = "none"
    """"Whether to include bias in LoRA layers"""


@dataclass
class PreprocessedCollator(DataCollatorMixin):
    """
    A minimal collator for datasets that are already tokenized and padded.
    """
    def __call__(self, features, return_tensors=None):
        """
        Dummy Collator for datasets that are already tokenized and padded.

        Args:
            features: A list of dictionaries containing the features for each sample.
                (each feature should have input_ids, attention_mask, labels as keys)
            return_tensors: Whether to return the batch as tensors or not. Defaults to None.

        Returns:
            A dictionary with the batched features.
        """
        batch = {}
        for k in list(features[0].keys()):
            if k in {"input_ids", "attention_mask", "labels", "weights"}:
                batch[k] = torch.stack([f[k] for f in features])
                
        return batch


class WeightedTrainer(SFTTrainer):
    """
    Trainer accepting wegihted samples
    """

    def compute_loss(self, model:PreTrainedModel, inputs:Dict[str, Any], return_outputs=False, num_items_in_batch=None):  
        """
        Computes the loss for a given model and inputs with optional weighting.

        Args:
            model (PreTrainedModel): The model to compute the loss with.
            inputs (Dict[str, Any]): The input data for the model, containing
                features like input_ids, attention_mask, and optionally 'weights'.
            return_outputs (bool, optional): If True, returns both the loss and the
                model outputs. Defaults to False.
            num_items_in_batch (int, optional): Number of items in the batch, used
                for adjusting loss scaling.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: The computed loss, and
            optionally the model outputs if return_outputs is True.

        Notes:
            - If 'weights' are provided in inputs, they are used to compute a
            weighted loss.
            - Applies CrossEntropyLoss with no reduction and adjusts the loss using
            the provided weights.
            - Supports model parallelism by ensuring the shifted labels and logits
            are on the same device.
            - Adjusts for tokens with label -100 by masking them in the loss.
            - Loss is averaged over non-zero elements and adjusted for device
            parallelism if enabled.
        """
        weights = inputs.pop("weights", None)
        
        if weights is None:
            outputs = model(**inputs)
            if return_outputs:
                return outputs.loss, outputs  # Return both loss and outputs for evaluation
            return outputs.loss

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        del outputs 

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels_ = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, self.model.vocab_size)
        shift_labels = shift_labels_.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        # flip back to (B, T)
        loss = loss.view(logits.shape[0], logits.shape[1]-1)
        # truncat the section of -100 labels 
        loss = loss.masked_fill(shift_labels_ == -100, 0.0)

        # adjust according to weights
        loss *= weights 
        # mean reduction by ignoreing masked elements
        loss = loss.sum()/(loss != 0).sum()

        del weights, shift_labels, shift_logits, shift_labels_, inputs
        
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    
    @dataclass
    class Args: 

        evaluate: bool = True
        """Whether to perform evaluation or not"""

        train:bool = True
        """Whether to perform training or not"""

        model_name:str = "microsoft/Phi-3.5-mini-instruct"
        """Name of the model to be loaded"""

        rft_data_path: str = "corrected-pred-parser-1.0lev-gsm8k_synthetic_data_747instances_5samples"
        """Path to the RFT dataset, sampled from the base model"""

        use_rft_data_only:bool = False
        """Whether to use only the RFT data for training or not"""

        use_weighted_training:bool = False
        """Whether to use higher weight on the RFT data for training or not"""

        run_name: str = ""
        """base name of the run"""

        seed:int = 42
        """Mannual Seed for the run"""

    
    args = tyro.cli(Args)
    if args.train:
        assert args.run_name, "Please provide a name for the training run"
        assert not args.train or not args.evaluate, "Cannot train and evaluate at the same time"

    seed_everything(args.seed)


    # --------------- Experiment Configs ---------------
    if args.rft_data_path and args.train:
        rft_data = datasets.load_from_disk(args.rft_data_path)
        print(f"RFT:\n")
        print(f"Loaded {len(rft_data)} instances from {args.rft_data_path}")
        rft_data = rft_data.rename_column("favored_solutions","answer")
        rft_data = rft_data.select_columns(["question", "answer"])
        args.run_name += "qlora_rft_"+args.rft_data_path.split("_")[-2]
    
    if not args.rft_data_path and args.train:
        print(f"SFT:\n")
        args.run_name += "qlora_sft"


    # --------------- Load Tokenizer ---------------
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    

    # --------------- Prepare Trainset ---------------
    train_dataset = datasets.load_dataset('gsm8k', 'main')['train']
    org_len = len(train_dataset)
    
    # We first remove the equation tags from the ground truth
    ## doing this since the base model cannot generate <<equation>> 
    ## and seems to perform reasoning correctly based on the natural language
    ## pattern effectively, we remove the equations from the ground truth
    ## to prevent potential confusion to the model during fine-tuning
    train_dataset = train_dataset.map(
        lambda x: GSM8KParser.remove_equations_from_gt(x["answer"])
    )
    if args.rft_data_path and args.train:
        if args.use_rft_data_only:
            # filter the original train dataset from the train dataset first 
            train_dataset = rft_data
        else:
            if args.use_weighted_training:
                # add normalised weights to the train dataset
                rft_weight = int(org_len / rft_data.shape[0])
                print(f"### applied weighted training on RFT data with unormalised weight: {rft_weight}")
                rft_data = rft_data.map(
                    lambda x : {"weights": torch.full((1, ), rft_weight)} 
                )
                train_dataset = train_dataset.map(
                    lambda x : {"weights": torch.full((1, ), 1)} 
                )
            
            train_dataset = filter_dataset(train_dataset, rft_data["question"], column_to_filter="question")
            train_dataset = datasets.concatenate_datasets([train_dataset, rft_data])
            assert len(train_dataset) == org_len, print(f"Expecting: {org_len}, Received: {len(train_dataset)}")

    TrainData = GSM8KDataset(train_dataset, tokenizer)
    

    # --------------- Test-Val Split ---------------
    TrainLoader = DataLoader(
        TrainData,
        batch_size=TrainingConfig.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=False, 
        persistent_workers=True 
    )
    print(f"Train size: {len(train_dataset)}")

    # split the test into val and test
    test_dataset = datasets.load_dataset('gsm8k', 'main')['test']
    test_dataset = test_dataset.map(
        lambda x: GSM8KParser.remove_equations_from_gt(x["answer"])
    )

    splitted_test = test_dataset.train_test_split(
        test_size=0.9, 
        seed= args.seed,
        shuffle=True,
    )

    # val data 
    val_dataset = splitted_test["train"]
    ValData = GSM8KDataset(val_dataset, tokenizer)
    ValLoader = DataLoader(
        ValData,
        batch_size=TrainingConfig.batch_size*2,
        shuffle=False,
        num_workers= os.cpu_count(),
    )
    print(f"Val size: {len(ValData)}")

    # test data
    test_dataset = splitted_test["test"]
    TestData = GSM8KDataset(test_dataset, tokenizer)
    print(f"Test size: {len(TestData)}")


    # --------------- Training Code ---------------
    if args.train:
        # quantization config 
        bb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # peft config
        lora_config = LoraConfig(
            target_modules = [
                "gate_proj", 
                "down_proj", 
                "up_proj",
                "k_proj", 
                "q_proj", 
                "v_proj", 
                "o_proj", 
            ],
            **LoraLayersConfig().__dict__,
        )

        # model
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            quantization_config = bb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

        ## TrainingArgs
        training_args = TrainingArguments(
            output_dir=f"models/{args.run_name}",
            per_device_train_batch_size=TrainingConfig.batch_size,
            per_device_eval_batch_size=TrainingConfig.batch_size*2,
            gradient_accumulation_steps=TrainingConfig.gradient_accum_steps,
            gradient_checkpointing=True,
            optim="adamw_8bit",
            save_strategy="epoch",
            learning_rate=TrainingConfig.learning_rate,
            bf16=True,
            tf32=False,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="wandb",
            eval_strategy="steps",
            eval_steps=TrainingConfig.val_interval,
            logging_steps=1,
            remove_unused_columns=False,
            num_train_epochs=TrainingConfig.max_n_epochs,
            
        )

        ## Perform Training 
        TrainerClass = WeightedTrainer if args.use_weighted_training else SFTTrainer
        trainer = TrainerClass(
            model=model,
            peft_config=lora_config,
            train_dataset=TrainData,
            eval_dataset=ValData,
            max_seq_length=TrainData.max_length,
            args=training_args,
            data_collator=PreprocessedCollator(),
            packing=False,
        )

        trainer.model.print_trainable_parameters()
        trainer.train()
        trainer.save_model(output_dir=f"models/{args.run_name}")
        print(f"### Model saved to models/{args.run_name} ###")


    # --------------- Evaluation Code ---------------
    if args.evaluate:

        if 'Phi-3.5' in args.model_name and not args.train:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-mini-instruct",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )

        print(f"#### Start evaluating {args.model_name} ####")
        model.to(torch.bfloat16)
        model.eval()
        model = torch.compile(model)

        test_loader = DataLoader(
            TestData,
            batch_size=32,
            shuffle=False,
            num_workers= os.cpu_count(),
        )

        maj_1s:List[float] = []
        questions:List[str] = []
        sampled_completions:List[str] = []
        ground_truth_completions:List[str] = []

        for i, batch in tqdm(enumerate(test_loader)):
            outstrings:List[str] = sample_answers(
                tokenizer,
                model,
                input_ids=batch["question_input_ids"],
                attention_mask=batch["question_attention_mask"],
                **GenerationConfig().__dict__,
            )
            preds = [GSM8KParser.get_answer_from_pred(pred)["answer_str_digit"] for pred in outstrings]
            labels = batch["answer_str_digit"]

            maj_1s.extend(
                [1 if pred == label else 0 for pred, label in zip(preds, labels)]
            )
            print(f"Moving avg: Batch {i} maj@1 -> {sum(maj_1s)/len(maj_1s)}")
            sampled_completions.extend(outstrings)
            ground_truth_completions.extend(batch["answer"])
            questions.extend(batch["question"])

        print(f"Maj@1: {sum(maj_1s)/len(maj_1s)}")
        df = pd.DataFrame.from_dict(
            {   
                "questions": questions,
                "sampled_completions": sampled_completions,
                "ground_truth_completions": ground_truth_completions,
                "maj_1s": maj_1s,
            }
        )
        out_path_ = f"results/{args.model_name.replace('/', '-')}_{args.run_name.replace('/', '-')}.csv"
        df.to_csv(out_path_, index=False)
        print(f"#### Results saved to {out_path_}")

