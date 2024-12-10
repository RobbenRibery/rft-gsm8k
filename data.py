from typing import List

import torch 
import datasets
from transformers import PreTrainedTokenizer
from typing import Dict, Any, Type

from torch.utils.data import Dataset
from utils import GSM8KParser
from prompt import EvalTemplate

def filter_dataset(dataset:datasets.Dataset, filter_list:List[str], column_to_filter:str="question") -> datasets.Dataset:
    """
    Filter out examples from a dataset that for the column_to_filter.
    Wae keep examples that are not in the filter_list
    
    Args:
        dataset: The dataset to filter
        filter_list: The list of questions to filter out
        column_to_filter: The column to filter on. Defaults to "question"
    
    Returns:
        The filtered dataset
    """
    filter_set = set(filter_list)

    # Filter function
    def should_keep(example):
        return example[column_to_filter] not in filter_set
    
    # Apply filter and save
    filtered_dataset = dataset.filter(should_keep)
    print(f"Kept {len(filtered_dataset)} non-overlapping instances")
    
    return filtered_dataset

def _apply_template(
    instance: Dict[str, Any],
    key: str,
    tokenizer: PreTrainedTokenizer,
    template:Type[EvalTemplate],
) -> Dict[str, str]:
    """
    Applies a chat template to an instance and formats it for tokenization.

    Args:
        instance (Dict[str, Any]): A dictionary containing the data instance,
            either a question or an answer.
        key (str): The key indicating the type of instance, could be "question"
            or "answer", depending on the dataset
        tokenizer (PreTrainedTokenizer): The tokenizer used to apply the chat
            template.
        template (Type[EvalTemplate]): The template class containing the
            system and user templates for formatting.

    Returns:
        Dict[str, str]: A dictionary with a single key-value pair where the key
        is "formatted_question" or "formatted_answer" based on the input key,
        and the value is the formatted and tokenized string.
        formatted question / answer are orginal input with prompt template applied
    """
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
        """
        Initialize the GSM8KDataset, which is an inheritance of torch.utils.data.Dataset

        Args:
        - dataset (datasets.Dataset): A dataset of GSM8K, which is a dataset of
          math problems in natural language. The dataset should have the following
          columns to begin with. The dataset could also contains additional columns.
          (i.e. number of hops, weigths for each instance etc.)
            - question (str): The question in natural language.
            - answer (str): The answer to the question.
        - tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing
          the dataset.

        The dataset is processed as follows:
        Step 1. The answer is parsed from a string to a dict with the following keys:
            - answer_str_digit (str): The answer as a string of digits.
        Step 2. The question is formatted by applying the prompt template to the question.
            - The answer is formatted by applying the prompt template to the answer.
        Step 3. The maximum length of the question, answer, and sequence is inferred.
        Step 4. The dataset is preprocessed by tokenizing the question and answer and
          creating the input_ids and attention_mask.
        Step 5. The dataset is set to the correct format for training.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer

        # parse answer
        self.dataset = self.dataset.map(
            lambda x: GSM8KParser.get_answer_from_pred(x["answer"])
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

        print(f"Maximum question num_tokens: {self.max_length_question}")
        print(f"Maximum answer num_tokens: {self.max_length_answer}")
        print(f"Maximum sequence num_tokens: {self.max_length}")

        # choose a maximum length at max_answer_length with 
        # 50 additional tokens
        self.inf_seq_length = self.max_length_answer + 50
        print(f"Maximum new tokens in generation: {self.inf_seq_length}")

        # Preprocess the dataset
        self.dataset = self.dataset.map(self._preprocess, batched=False)

        # columns which must be output in the tensors format
        tensors_cols = [
            "input_ids",
            "attention_mask",
            "question_input_ids",
            "question_attention_mask",
            "labels",
        ]
        # we don't process weights here, refers to lora.py 
        if "weights" in self.dataset.column_names:
            tensors_cols.append("weights")
        
        self.dataset.set_format(
            type="pt",
            columns=tensors_cols,
            output_all_columns=True,
        )
        print(f"Setup Completed dataset:\n{self.dataset}")

    def _left_pad_tensor(self, tensor: torch.Tensor, padding_length: int, pad_value: int) -> torch.Tensor:
        """Left-pad a tensor with a specified value."""
        if padding_length == 0:
            return tensor
        return torch.cat([
            torch.full((padding_length,), pad_value, dtype=tensor.dtype),
            tensor
        ])
    
    def _right_pad_tensor(self, tensor: torch.Tensor, padding_length: int, pad_value: int) -> torch.Tensor:
        """Left-pad a tensor with a specified value."""
        if padding_length == 0:
            return tensor
        return torch.cat([
            tensor,
            torch.full((padding_length,), pad_value, dtype=tensor.dtype),
        ])
    
    def _create_labels_tensor(self, padding_length: int, question_length: int, answer_ids: torch.Tensor) -> torch.Tensor:
        """Creates labels tensor with proper padding and masking."""
        return torch.cat([
            torch.full((question_length,), -100),
            answer_ids,
            torch.full((padding_length,), -100),
        ])
    
    def _preprocess(self, instance: Dict[str, str|torch.Tensor]):
        """
        Preprocesses a single instance of the dataset.

        Args:
            instance: A dict containing the following keys:
                - formatted_question: The formatted question string.
                - formatted_answer: The formatted answer string.

        Returns:
            A dict of preprocessed tensors with the following keys:
                - question_input_ids: The left-padded question input ids.
                - question_attention_mask: The left-padded question attention mask.
                - input_ids: The left-padded sequence input ids.
                - attention_mask: The left-padded sequence attention mask.
                - labels: The left-padded labels for fine-tuning only on completions.
        """
        out = {}
        question, answer = instance["formatted_question"], instance["formatted_answer"]

        # tokenize formatted question
        question_encodings = self.tokenizer(
            question, return_tensors="pt", truncation=True
        )

        # lef pad the question encodings manually (this is for inference)
        q_padding_length = self.max_length_question - question_encodings["input_ids"].shape[1]
        out["question_input_ids"] = self._left_pad_tensor(
            question_encodings["input_ids"].squeeze(0),
            q_padding_length,
            self.tokenizer.pad_token_id
        )
        out["question_attention_mask"] = self._left_pad_tensor(
            question_encodings["attention_mask"].squeeze(0),
            q_padding_length,
            0
        )

        # tokenizer formatted answer
        answer_encodings = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
        )

        # concatenate the question and the answer input_ids
        seq_ids = torch.cat(
            [
                question_encodings["input_ids"].squeeze(0),
                answer_encodings["input_ids"].squeeze(0),
            ],
            dim=0,
        )

        # left pad the input_ids for the entire sequence manually
        sequence_padding_length = self.max_length - seq_ids.shape[0]
        if sequence_padding_length > 0:
            # right padding with self.tokenizer.pad_token_id
            padded_seq_ids = self._right_pad_tensor(
                seq_ids,
                sequence_padding_length,
                self.tokenizer.pad_token_id,
            )
        elif sequence_padding_length < 0:
            raise ValueError(
                f"Sequence is too long: Received length{seq_ids.shape[0]} > Mazimum length {self.max_length}"
            )
        else:
            padded_seq_ids = seq_ids
        out["input_ids"] = padded_seq_ids

        # right pad the attention_mask for the entire sequence manually
        # cat[ pad span(0) | seq_ids(1) ]
        attention_mask = self._right_pad_tensor(
            torch.ones(seq_ids.shape[0]), 
            sequence_padding_length, 
            0,
        )
        out["attention_mask"] = attention_mask
        
        # left pad the labels for fine-tuning only on completions 
        # cat[ pad span(-100) | quesiton_input_ids(-100) |  answer_input_ids(-100) ]
        labels = self._create_labels_tensor(
            sequence_padding_length,
            question_encodings["input_ids"].shape[1],
            answer_encodings["input_ids"].squeeze(0)
        )
        out["labels"] = labels
        
        return out

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.dataset[idx]