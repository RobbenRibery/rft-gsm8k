import datasets
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

from typing import List, Tuple
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.generation import GenerateDecoderOnlyOutput

from data import GSM8KDataset
from tqdm import tqdm 
from utils import (
    INVALID_ANSWER,
    GSM8KParser,
    save, 
    load,
)
from nltk import edit_distance
from os import cpu_count 
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

WRONG_SOLUTIONS_PLACEHOLDER = "<NOWRONG SOLUTION>"

def _socre_equations(equations: List[str], completions:List[str]) -> Tuple[int, int, int]:
    """
    Scores the equations based on the pair-wise levenshtein distance
    and the number of lines in each equation. 

    The score is a normalised average of the two, with the levenshtein
    distance normalised by the maximum levenshtein distance, and the
    number of lines normalised by the maximum number of lines.

    The function returns a tuple of three integers: the index of the
    best solution, the index of the worst solution, and the gap between
    the two.
    """
    if len(equations) == 1:
        return 0, 0, 0

    # iterate through the equations
    # compute the pair-wise levenshtein distance
    # note that we fill d[i][j] into d[j][i] since it is symmetric
    distances = defaultdict(int)
    for i in range(len(equations)):
        for j in range(i + 1, len(equations)):
            d_ij = edit_distance(equations[i], equations[j])
            distances[i] += d_ij
            distances[j] += d_ij # back fill the lower triagnular-section

    # normalise the maximum levenshtein distance
    max_dist = max(distances.values())
    if max_dist == 0:
        max_dist = 1
    
    # ge the number of lines and normaliser as well
    nlines = [len(comp.split("\n")) for comp in completions]
    max_nlines = max(nlines)
    if max_nlines == 0:
        max_nlines = 1
    
    for idx, score in distances.items():
        # here, we take the average of the levenshtein distance and the number of lines
        # since our equaiton parser is not perfect, we use the number of lines as well
        # as another proxy for the novelty of the overall solution
        # one could further tune such hyperparameters to make it more robust
        distances[idx] = 0.5 * (score/max_dist) + 0.5 * nlines[idx]/max_nlines

    max_idx = max(distances, key = distances.get)
    min_idx = min(distances, key = distances.get)

    return max_idx, min_idx, distances[max_idx]-distances[min_idx]

@torch.no_grad()
def get_generations(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    dataset:GSM8KDataset,
    batch_size:int=4,
    **kwargs,
) -> List[torch.Tensor]:  
    
    # same as sample_answers, but only perform the gpu operations
    assert tokenizer.padding_side == "left"
    assert kwargs["return_dict_in_generate"]
    results = []

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers= cpu_count(),
        shuffle=False,
        drop_last=False,
    )

    for instances in tqdm(data_loader):
        input_ids: torch.Tensor = instances["question_input_ids"].to(model.device)
        attention_mask: torch.Tensor = instances["question_attention_mask"].to(model.device)
        
        generation_outs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            **kwargs
        )
        results.append(generation_outs.sequences.cpu())
        
        del input_ids, instances, attention_mask, generation_outs
        torch.cuda.empty_cache()
        #break 
    return results


def _filter_completions(
    sequences:torch.Tensor, 
    input_length:int,
    tokenizer:PreTrainedTokenizer
    ) -> Tuple[List, List, List]: 
    
    # get the new tokens from generation
    new_tokens = sequences[:, input_length :]
    completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # go through each completion and return the best and the worst tuple
    incorrect_completions:List[str] = [] #incorrect completion tokens
    unique_correct_completions:List[str] = [] # correct completion tokens 
    correct_completions_visited = set() # visited correct completions
    unique_correct_completions_eqs:List[str] = [] # reasoning traces/equations

    for _, completion in enumerate(completions):
        
        # parse the answer
        answer = GSM8KParser.get_answer_from_pred(completion)
        
        # filter based on correcteness
        if answer["answer_str_digit"] == INVALID_ANSWER:
            incorrect_completions.append(completion)
        else:
            # add only if not visited
            if completion not in correct_completions_visited:
                correct_completions_visited.add(completion)
                unique_correct_completions.append(completion)

                # parse equations
                equaiton_str = " ".join(
                    GSM8KParser.parse_equations_from_pred(completion)["equations"]
                )
                unique_correct_completions_eqs.append(equaiton_str)
    
    return unique_correct_completions, incorrect_completions, unique_correct_completions_eqs


def generate_synthetic_data(
    model:PreTrainedModel, 
    tokenizer:PreTrainedTokenizer, 
    dataset: GSM8KDataset, 
    name:str = "gsm8k",
    batch_size:int = 4,
    **generation_config
) -> datasets.Dataset:

    questions:List[str] = []
    favored_solutions: List[str] = []
    infavored_solutions: List[str] = []
    wrong_solutions: List[str] = []
    gaps:List[int] = []
    
    # get the model generations
    generations:List[torch.Tensor] = get_generations(
        tokenizer, 
        model, 
        dataset, 
        batch_size=batch_size,
        **generation_config
    )
    save(f"generations/{name}_generations", generations)

    # split generations to per instance level
    sequences = []
    for batch_gen in generations: 
        batch_sequences = batch_gen

        for i in range(0, batch_sequences.shape[0], generation_config["num_return_sequences"]):
            sequences.append(batch_sequences[i:i+generation_config["num_return_sequences"]])
            assert len(sequences[-1]) == generation_config["num_return_sequences"]

    # save the generations in case we loose it later
    save(f"generations/{name}_sequences", sequences)

    # iterate through generations for each instances 
    for i, instance_sequences in enumerate(sequences):

        unique_correct_completions, \
        incorrect_completions, \
        unique_correct_completions_eqs = _filter_completions(
            instance_sequences,
            dataset.max_length_question,
            tokenizer,
        )

        # go to next instance if all failed
        if not unique_correct_completions:
            print(f"No correct completions for istance {i}")
            continue 
        
        # measure the utiliy of each completion according to their rationale
        ## one creteria could be the novelty of the equations
        questions.append(dataset[i]["question"])
        best_idx, best_worst_idx, gap = _socre_equations(
            unique_correct_completions_eqs,
            unique_correct_completions,
        )
        # best_idx: idx of the completion containing the most novel equation sets 
        # best_worst_idx: idx of the completion containing the least novel equation sets 
        # gap between the novelty metric 

        # pick favored and infavored completions
        favored_completion = unique_correct_completions[best_idx]
        infavored_completion = unique_correct_completions[best_worst_idx]
        
        # since levenstein distance is a hard metric and our parse is not super reliable, 
        # in case of tie, we pick the longest to be the best:
        if best_idx == best_worst_idx and len(unique_correct_completions) > 1:
            unique_correct_completions.sort(key=len, reverse=True)
            favored_completion = unique_correct_completions[0]
            infavored_completion = unique_correct_completions[-1]
            gap = len(unique_correct_completions[0]) - len(unique_correct_completions[-1])

        favored_solutions.append(favored_completion)
        infavored_solutions.append(infavored_completion)
        gaps.append(gap)

        # pick the wrong completions
        if not incorrect_completions:
            wrong_solutions.append(WRONG_SOLUTIONS_PLACEHOLDER) 
            # place holder in case all correct
        else:
            wrong_solutions.append(incorrect_completions[0])
    
    dataset = datasets.Dataset.from_dict(
        {
            "question": questions,
            "favored_solutions": favored_solutions,
            "infavored_solutions": infavored_solutions,
            "wrong_solutions": wrong_solutions,
            "favored_infavored_gaps":gaps,
        }
    )
    return dataset


if __name__ == "__main__":

    # load the model
    MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
    model:PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # get the train dataset
    train_dataset = datasets.load_dataset('gsm8k', 'main')['train']
    # we use the number of hops to filter the difficulty of the instances
    # and we want our rejection sampling to focus on those part more 
    # (due to limited computational power)
    train_dataset = train_dataset.map(
        lambda x: GSM8KParser.get_num_hops(x['answer'])
    )
    train_dataset = train_dataset.sort(column_names="num_hops", reverse=True)

    @dataclass
    class SyntheticDataConfig:
        temperature: float = 0.7
        num_return_sequences: int = 5
        batch_size:int = 2
        reject_sampling_percentage: float = 0.1
        num_reject_samples: int = int(len(train_dataset)*reject_sampling_percentage)
        run_name = f"gsm8k_synthetic_data_{num_reject_samples}instances_{num_return_sequences}samples"

    # select the top-K instances for rejection sampling
    train_dataset = train_dataset.select(
        range(
            int(
                len(train_dataset)*SyntheticDataConfig.reject_sampling_percentage
            )
        )
    )
    TrainData = GSM8KDataset(train_dataset, tokenizer)
    print(f"Selected {len(train_dataset)} instances for rejection sampling out of {len(train_dataset)}")
    print(f"Each instance will generate {SyntheticDataConfig.num_return_sequences} completions")

    # Generation Config 
    generation_config = {
        "max_new_tokens" : TrainData.inf_seq_length,
        "temperature": SyntheticDataConfig.temperature,
        "num_return_sequences": SyntheticDataConfig.num_return_sequences,
        "eos_token_id":tokenizer.eos_token_id,  # Specify the EOS token
        "pad_token_id":tokenizer.eos_token_id, 
        "do_sample":True,
        "output_scores":False,
        "return_dict_in_generate":True,
    }

    # compile to further accelerate inference 
    model.eval()
    model = torch.compile(model)

    # generate the dataset
    dataset = generate_synthetic_data(
        model, 
        tokenizer, 
        TrainData,
        name=SyntheticDataConfig.run_name,
        batch_size=SyntheticDataConfig.batch_size,
        **generation_config,
    )
    
    # save the dataset
    dataset.save_to_disk(SyntheticDataConfig.run_name)