import torch 
import datasets
from transformers import PreTrainedTokenizer
from typing import Dict, Any, Type

from torch.utils.data import Dataset
from utils import GSM8KParser
from prompt import EvalTemplate

def _apply_template(
    instance: Dict[str, Any],
    key: str,
    tokenizer: PreTrainedTokenizer,
    template:Type[EvalTemplate],
) -> Dict[str, str]:
    if key == "question":
        content = [
            {"role": "system", "content": template.system},
            {
                "role": "user",
                "content": template.user.format(
                    question=instance[key]
                ),
            },
        ]
        add_generation_prompt = True
    elif key == "answer":
        content = [{"role": "assistant", "content": instance["answer"]}]
        add_generation_prompt = False

    formmated_str = tokenizer.apply_chat_template(
        content,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if key == "answer":
        # remove the first <|assistant|> from the generated prompt 
        # since it's part of the question already  
        formmated_str = formmated_str.lstrip("<|assistant|>").lstrip()

    return {f"formatted_{key}": formmated_str}

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
            lambda x: _apply_template(x, "question", self.tokenizer, EvalTemplate)
        )
        # format answer
        self.dataset = self.dataset.map(
            lambda x: _apply_template(x, "answer", self.tokenizer, EvalTemplate)
        )

        # infer max length
        self.dataset = self.dataset.map(
            lambda x: GSM8KParser.get_question_length(x["formatted_question"], tokenizer)
        )
        self.dataset = self.dataset.map(
            lambda x: GSM8KParser.get_answer_length(x["formatted_answer"], tokenizer)
        )

        self.max_length_question = max(self.dataset["question_length"])
        self.max_length_answer = max(self.dataset["answer_length"])
        self.max_length = self.max_length_question + self.max_length_answer

        print(f"Maximum answer num_tokens: {self.max_length_answer}")
        print(f"Maximum question num_tokens: {self.max_length_question}")
        print(f"Maximum sequence num_tokens: {self.max_length}")

        # should not hard code here, but choose a large enough sequence length
        # during sampling, for now. 
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

        question, answer = instance["formatted_question"], instance["formatted_answer"]

        # tokenize formatted question
        question_encodings = self.tokenizer(
            question, return_tensors="pt", truncation=True
        )

        # lef pad the question encodings manually (this is for inference)
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

        # left pad the input_ids for the entire sequence manually
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

        # left pad the attention_mask for the entire sequence manually
        # cat[ pad span(0) | seq_ids(1) ]
        attention_mask = torch.concat(
            [
                torch.tensor([0]).repeat(diff),
                torch.ones(seq_ids.shape[0]),
            ],
            dim=0,
        )
        assert out["input_ids"].shape[0] == attention_mask.shape[0]
        out["attention_mask"] = attention_mask
        
        # left pad the labels for fine-tuning only on completions 
        # cat[ pad span(-100) | quesiton_input_ids(-100) |  answer_input_ids(-100) ]
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

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.dataset[idx]