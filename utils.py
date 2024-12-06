import re
import datasets
import torch

from collections import Counter
from typing import List, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
import pickle 

INVALID_ANSWER = "<INVALID_ANSWER>"
VALID_ANSWER_CHARS = set([str(i) for i in range(10)] + [",", ".", "-"])


def save(filename:str, obj):
    """
    Saves a python object to a pickle file.

    Args:
        filename: The filename of the pickle file to save to.
        obj: The python object to save.
    """
    with open(f'{filename}.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {filename}.pickle")

def load(filename:str):
    """
    Loads a pickled file and returns the unpickled object.

    Args:
        filename: The filename of the pickled file to load.

    Returns:
        The unpickled object.
    """
    with open(f'{filename}.pickle', 'rb') as handle:
        return pickle.load(handle)

def inspect_instance(data:Dict[str,str], idx: int) -> None:
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
    input_ids: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    **kwargs,
) -> List[str]:
    """
    Samples answers from a given model and tokenizer.

    Args:
        tokenizer: The tokenizer to use.
        model: The model to use.
        chats: The list of chats to sample answers from.
        input_ids: The input ids to use. If given, `chats` will be ignored.
        attention_mask: The attention mask to use. If given, `chats` will be ignored.
        **kwargs: Additional keyword arguments to pass to `model.generate`.

    Returns:
        A list of strings, where each string is an answer sampled from the model.
    """
    assert tokenizer.padding_side == "left"
    assert kwargs["return_dict_in_generate"]

    if chats:
        encodings: torch.Tensor = tokenizer.batch_encode_plus(
            chats,
            return_tensors="pt",
            padding="longest",
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

    generation_out = model.generate(
        input_ids=input_ids.to(model.device), 
        attention_mask=attention_mask.to(model.device), 
        **kwargs
    )

    new_tokens = generation_out.sequences[:, input_ids.shape[1] :]

    del input_ids, attention_mask, generation_out
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


class GSM8KParser:

    @classmethod
    def get_question_length(
        cls, question_text: str, tokenizer: PreTrainedTokenizer
    ) -> Dict[str, int]:
        """Get the number of tokens in the question"""
        return {"question_length": len(tokenizer(question_text)["input_ids"])}

    @classmethod
    def get_answer_length(
        cls, answer_text: str, tokenizer: PreTrainedTokenizer
    ) -> Dict[str, int]:
        """Get the number of tokens in the answer"""
        return {"answer_length": len(tokenizer(answer_text)["input_ids"])}

    @classmethod
    def get_answer_from_gt(cls, answer_text: str) -> Dict[str, str]:
        """
        This function is strict that it will gurantee to find a 
        valid answer in the given answer_text, provided that the answer
        text from the GSM8K Dataset (not generated answer)
        Any violation of the format will wiae an error 

        Parse the answer from the ground truth format.

        The ground truth format is a single string with the following rules:

        1. The last line should start with "####"
        2. The last line should contain only digits

        Args:
            answer_text (str): The answer text from the GMS8K Dataset

        Returns:
            A dictionary with a single key "answer_str_digit" and the
            corresponding value as the digit-only answer string.
        """
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

        # last line 
        pred_answer_text = pred_answer_text.strip().split("\n")[-1]

        # Pattern explanation:
        # ####\s*               Match '####' followed by optional whitespace
        # (                     Start capturing group
        #   -?                  Optional matching negative to start with 
        #   \d+                 One or more digits
        #   (?:                 Start non-capturing group for optional separator and more digits
        #     (?:               Start non-capturing group for separator alternatives
        #       (?<!\,)\,(?!\,) Single comma with no commas adjacent
        #       |          - OR
        #       (?<!\.)\.(?!\.) Single period with no periods adjacent
        #     )                 End separator alternatives group
        #     \d+               One or more digits
        #   )*                  Allow multiple separator-number pairs
        # )                     End main capturing group
        pattern = r"####\s*(-?\d+(?:(?:(?<!\,)\,(?!\,)|(?<!\.)\.(?!\.))\d+)*)"

        match_ = re.search(pattern, pred_answer_text)
        if not match_:
            return {"answer_str_digit": INVALID_ANSWER}

        candidate:str = match_.group(1)
        candidate_wihout_comma = re.sub(r"(?<!\,)\,(?!\,)", "", candidate).strip()

        try:
            eval(candidate_wihout_comma)
            return {"answer_str_digit": candidate_wihout_comma}
        except Exception:
            return {"answer_str_digit": INVALID_ANSWER}

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
    def remove_equations_from_gt(cls, answer: str) -> Dict[str, List[str]]:
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
        pattern = r"<<.*?>>"
        return {"answer": re.sub(pattern, "", answer)}

    @classmethod
    def parse_equations_from_pred(cls, text: str, include_text: bool=False) -> List[str]:
        """
        Parse the equations from a string of text generated by the model.

        Parameters
        ----------
        text : str
            The string of text to extract equations from
        include_text : bool
            Whether to include the text in the equation or not

        Returns
        -------
        List[str]
            The list of equations extracted from the text
        """
        equations = []

        def extract_digits(text_with_digits: str) -> str:
            """
            Extracts only the digits from a string.

            Args:
                text_with_digits (str): string that contains digits.

            Returns:
                str: a string with only the digits
            """
            return "".join(
                [
                    c
                    for c in text_with_digits
                    if c.isdigit() or c in {"+", "-", "*", "/", "="}
                ]
            )

        # eq pattern consistes of two parts:
        # 1. left side: matching the digits(continuous) + any chracters
        # 2. right side: matching the any character (continuous) + digits
        # 3. repeat the right side pattern indefinately
        # pattern explanation 
        if include_text:
            eq_pattern = r"\S*?(\d.*?\d+.*?)=(.*?\d+)+\S*?"
        else:
            eq_pattern = r"(\d+.*?)=(.*?\d+)+"

        regex = re.compile(eq_pattern)
        matches = list(regex.finditer(text))
        if not matches:
            return {"equations": equations}

        for _, m in enumerate(matches):
            start, end = m.start(), m.end()
            matched_string = text[start:end]
            if not include_text:
                matched_string = extract_digits(matched_string)
            equations.append(matched_string)

        # dedup equations 
        # we would like to make sure that there is no duplication of equations
        # within the same completion (LLM does repeat sometime)
        equations = list(set(equations))
        return {"equations": equations}


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

    ### Added a few test cases for you to play with 
    ## =>=> python utils.py

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
    #### 123
    #### 78,000,000,,
    """
    assert GSM8KParser.get_answer_from_gt(ex3) == {"answer_str_digit": "78000000"}
    assert GSM8KParser.get_answer_from_pred(ex3) == {
        "answer_str_digit": "78000000"
    }, print(GSM8KParser.get_answer_from_pred(ex3))

    ex4 = """
    # 1+1 = 3 
    #### perform calculation 
    #### here is the final answer
    #### 78.5023419872983 

    
    """
    assert GSM8KParser.get_answer_from_gt(ex4) == {"answer_str_digit": "78.5023419872983"}
    assert GSM8KParser.get_answer_from_pred(ex4) == {"answer_str_digit": "78.5023419872983"}, print(
        GSM8KParser.get_answer_from_pred(ex3)
    )

    text = """
    3 + 9 = 12 
    4 + 7 = 8 (somthing)

    4 apple + 8 apple = 12 apple 

    1+2-3=1+3=-2=4

    $4 * $5 + $6 * $7 = 12
    """
    out = GSM8KParser.parse_equations_from_pred(text, include_text=False)
    print(f"Eqautions without text\n{out}")
    out = GSM8KParser.parse_equations_from_pred(text, include_text=True)
    print(f"Eqautions with text\n{out}")
