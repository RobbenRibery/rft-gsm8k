import datasets
import torch

from typing import Dict, List, Any
from collections import defaultdict
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.generation import GenerateDecoderOnlyOutput

from data import GSM8KDataset
from tqdm import tqdm 
from utils import (
    INVALID_ANSWER,
    GSM8KParser,
    sample_answers,
    parse_equations,
)
from nltk import edit_distance

def _argmaxmin_levenshtein_distance(equations: List[str]) -> str:
    
    distances = defaultdict(int)
    for i in range(len(equations)):
        for j in range(i + 1, len(equations)):
            d_ij = edit_distance(equations[i], equations[j])
            distances[i] += d_ij
            distances[j] += d_ij # back fill the lower triagnular-section

    max_idx = max(distances, key = distances.get)
    min_idx = min(distances, key = distances.get)

    return equations[max_idx], equations[min_idx], equations[max_idx]-equations[min_idx]

@torch.no_grad()
def get_generations(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    dataset:GSM8KDataset,
    **kwargs,
) -> List[GenerateDecoderOnlyOutput]:  
    
    # same as sample_answers, but only perform the gpu operations
    assert tokenizer.padding_side == "left"
    assert kwargs["return_dict_in_generate"]
    results = []

    for i in tqdm(range(len(dataset))):
        instnace = dataset[i]
        input_ids: torch.Tensor = instnace["question_input_ids"]
        attention_mask: torch.Tensor = instnace["question_attention_mask"]
        generation_out = model.generate(
            input_ids=input_ids.to(model.device), 
            attention_mask=attention_mask.to(model.device), 
            **kwargs
        )
        results.append(generation_out)
    return results


def generate_synthetic_data(
    model:PreTrainedModel, 
    tokenizer:PreTrainedTokenizer, 
    dataset: GSM8KDataset, 
    **generation_config
) -> datasets.Dataset:

    accepted_solution: List[str] = []
    rejected_solution: List[str] = []

    for instance in dataset:
        
        # get the inputs
        input_ids: torch.Tensor = instance["input_ids"]
        attention_mask: torch.Tensor = instance["attention_mask"]

        # return new tokens
        completions: List[str] = sample_answers(
            tokenizer,
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )

        instance_incorrect_completions:List[str] = [] #incorrect completion tokens
        unique_instance_correct_completions:List[str] = [] # correct completion tokens 
        instance_correct_completions_visited = set()

        completion_equations:List[str] = [] # reasoning traces 

        for completion in enumerate(completions):
            # parse the answer
            answer = GSM8KParser.get_answer_from_pred(completion)
            
            # filter based on correcteness
            if answer["answer_str_digit"] == INVALID_ANSWER:
                instance_incorrect_completions.append(completion)
            else:
                # add only if not visited
                if completion not in instance_correct_completions_visited:
                    instance_correct_completions_visited.add(completion)
                    unique_instance_correct_completions.append(completion)

                    equaitons:List[str] = parse_equations(completion)
                    equaiton_str = " ".join(equaitons)

                    completion_equations.append(equaiton_str)

        # go to next instance if all failed
        if not unique_instance_correct_completions:
            continue 
        
        # pick the best completion
        ## one creteria could be the novelty of the equations
        ## hence we pick the unique_instance_correct_completion 
        ## which has the maximum sum of the levelenshtein distance
        ## to its counterparts
        best_completion, worst_correct_completion = \
            _argmaxmin_levenshtein_distance(unique_instance_correct_completions)
        

        # pick the worst completion
        if not instance_incorrect_completions:
            worst_completion = worst_correct_completion
        else:
            worst_completion = instance_incorrect_completions[-1]

        accepted_solution.append(best_completion)
        rejected_solution.append(worst_completion)


    return datasets.Dataset.from_dict(
        {
            "question": dataset["question"],
            "accepted_solution": accepted_solution,
            "rejected_solution": rejected_solution,
        }
    )
