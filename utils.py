from typing import List, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

from collections import Counter
import re
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations

INVALID_ANSWER = "<INVALID_ANSWER>"
VALID_ANSWER_CHARS = set([str(i) for i in range(10)] + [",", ".", "-"])


def inspect_instance(data, idx: int) -> None:
    """
    Prints out the key-value pairs of a given instance in a dataset at idx.

    Args:
        data: The dataset to inspect.
        idx: The index of the instance to inspect.
    """
    for k, v in data[idx].items():
        print(f"{k}\n{v}")
    print("*" * 50)



@torch.no_grad()
def sample_answers(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    chats: List[str] = None,
    input_ids:torch.Tensor = None,
    attention_mask:torch.Tensor = None,
    **kwargs,
) -> List[str]:
    assert tokenizer.padding_side == "left"
    assert kwargs["return_dict_in_generate"]
    
    if chats:
        encodings: torch.Tensor = tokenizer.batch_encode_plus(
            chats,
            return_tensors="pt",
            padding="longest",
        )
        input_ids = encodings["input_ids"].cuda()
        attention_mask = encodings["attention_mask"].cuda()

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, 
        enable_math=False,
        enable_mem_efficient=False
    ):
        generation_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    new_tokens = generation_out.sequences[:, input_ids.shape[1]:] #[G, Tc,]

    del input_ids
    del attention_mask

    if kwargs["output_scores"] and kwargs["num_return_sequences"] > 1:
        new_token_probs = torch.stack(generation_out.scores, dim=1).softmax(-1) #[G, Tc, vocab_size]
        
        answer_logprobs_norm = torch.log(
            torch.gather(
                new_token_probs, 
                dim=2, 
                index=new_tokens[:,:,None]
            ).squeeze(-1) #[G, Tc]
        ).mean(-1) #[G, Tc]
        # this step compute the mean of log(p(yt|y_1...y_{t-1},x)) over all sampled completions
        # It's gonna be a negative value, but the higher the better
        # this is not perfect, we should ignore the <eos> token that's padded across sampled completions
        del new_token_probs

        ranks = torch.argsort(answer_logprobs_norm, descending=True) #[G, Tc]
        del answer_logprobs_norm

        new_tokens = new_tokens[ranks]
        del ranks 

    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True) 


class GSM8KParser:

    @classmethod
    def get_question_length(cls, question_text: str, tokenizer:PreTrainedTokenizer) -> Dict[str, int]:
        return {"question_length": len(tokenizer(question_text)["input_ids"])}
    
    @classmethod
    def get_answer_length(cls, answer_text: str, tokenizer:PreTrainedTokenizer) -> Dict[str, int]:
        return {"answer_length": len(tokenizer(answer_text)["input_ids"])}

    @classmethod
    def get_answer_from_gt(cls, answer_text: str) -> Dict[str, str]:
        lines = answer_text.strip().split("\n")

        if "####" not in lines[-1]:
            raise ValueError(f"Ill-formed answer provided: {answer_text}")

        answer_str: str = lines[-1].replace("####", "").strip()
        answer_str_digit = answer_str.replace(",", "")

        try:
            eval(answer_str_digit)
        except Exception as e:
            raise ValueError(f"Ill-formed answer provided: {answer_str}") from e

        return {"answer_str_digit": answer_str_digit}

    @classmethod
    def get_answer_from_pred(cls, pred_answer_text: str) -> Dict[str, str]:

        # positive lookahead, terminate at the end of the string or the next "####"
        answer_pattern = r"(####.*?)(?=\Z|####)"
        matches: List[str] | None = re.findall(
            answer_pattern, pred_answer_text, flags=re.DOTALL
        )
        if not matches:
            return {"answer_str_digit": INVALID_ANSWER}

        last_match = matches[-1].replace("#", "").strip()
        last_match = re.sub(r"(?<!\,)\,(?!\,)", "", last_match)

        # forward search to cover all digits after ####
        candidate = ""
        for i, c in enumerate(last_match):

            if i == 0 and c == "-":
                candidate += c
                continue

            if c in VALID_ANSWER_CHARS:
                try:
                    eval(candidate + c)
                    candidate += c
                except Exception:
                    break
            else:
                break

        if not candidate:
            return {"answer_str_digit": INVALID_ANSWER}

        return {"answer_str_digit": candidate}

    @classmethod
    def get_num_hops(cls, answer_text: str) -> Dict[str, int]:
        """
        Calculate the number of steps in the solution.
        since GMS8k is highly structured.
        The higher the ouput, the more complex the problem

        Parameters
        ----------
        answer_text : str
            The answer text

        Returns
        -------
        int
            The number of steps in the solution
        """
        return {"num_hops": len(answer_text.strip().split("\n")) - 1}

    @classmethod
    def extract_equations(text: str) -> Dict[str, List[str]]:
        """
        Extract list of equations from a string of text.

        Parameters
        ----------
        text : str
            The string of text to extract equations from

        Returns
        -------
        List[str] | None
            The list of equations extracted from the text, or None if no equations were found
        """
        pattern = r"<<(.+?)>>"
        list_of_eqs = re.findall(pattern, text)
        if not list_of_eqs:
            raise ValueError("No equations found")

        return {"equations": list_of_eqs}


class GMS8KEvaluator:

    def __init__(self):
        pass

    def _get_maj(self, candidates: List[str]) -> str:
        """
        Get the majority vote from a list of predictions.

        Parameters
        ----------
        candidates : List[float|int]
            A list of predictions. -1 is used to indicate invalid predictions.
            -1 is invalid answers

        Returns
        -------
        str
            The majority vote. -1 if all predictions are invalid.
        """
        valid_answers = [pred for pred in candidates if pred != INVALID_ANSWER]
        if not valid_answers:
            return INVALID_ANSWER

        counts = Counter(valid_answers)
        return max(counts, key=counts.get)

    def get_maj_at_k(self, candidates: List[str], answer: str) -> int:
        """
        Evaluate a list of predictions and return the accuracy.

        Parameters
        ----------
        candidates : List[str]
            A list of predictions.
        answer : str
            The correct answer.

        Returns
        -------
        int
            1 if the majority vote is correct, 0 otherwise.
        """
        if isinstance(candidates, str):
            candidates = [candidates]
        return int(self._get_maj(candidates) == answer)


if __name__ == "__main__":

    ex1 = """
    Together, they made 80+100 = <<80+100=180>>180 pizzas on the second day.
    In the two days, the two of them made a total of 180 +200 = <<180+200=380>>380 pizzas.
    #### 380.00 
    """

    assert GSM8KParser.get_answer_from_gt(ex1) == {"answer_str_digit": "380.00"}
    assert GSM8KParser.get_answer_from_pred(ex1) == {
        "answer_str_digit": "380.00"
    }, print(GSM8KParser.get_answer_from_pred(ex1))

    ex2 = """
    # 1+1 = 3 
    # 3 + 2 = 5 
    # so bla bla bla
    #### -789,678,787,878
    """

    assert GSM8KParser.get_answer_from_gt(ex2) == {"answer_str_digit": "-789678787878"}
    assert GSM8KParser.get_answer_from_pred(ex2) == {
        "answer_str_digit": "-789678787878"
    }, print(GSM8KParser.get_answer_from_pred(ex2))

    ex3 = """
    # 1+1 = 3 
    # 3 + 2 = 5 
    # so bla bla bla
    #### -78,000,000
    """
    assert GSM8KParser.get_answer_from_gt(ex3) == {"answer_str_digit": "-78000000"}
    assert GSM8KParser.get_answer_from_pred(ex3) == {
        "answer_str_digit": "-78000000"
    }, print(GSM8KParser.get_answer_from_pred(ex3))

    ex4 = """
    # 1+1 = 3 
    #### perform calculation 
    #### here is the final answer
    #### -78 
    """
    assert GSM8KParser.get_answer_from_gt(ex4) == {"answer_str_digit": "-78"}
    assert GSM8KParser.get_answer_from_pred(ex4) == {"answer_str_digit": "-78"}, print(
        GSM8KParser.get_answer_from_pred(ex3)
    )
