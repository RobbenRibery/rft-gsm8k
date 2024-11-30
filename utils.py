from typing import List, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

from collections import Counter 
import re 
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations

INVALID_ANSWER = '<INVALID_ANSWER>' 
VALID_ANSWER_CHARS = set(
    [str(i) for i in range(10)] + [',', '.', '-']
)

def inspect_instance(data, idx:int) -> None:
    """
    Prints out the key-value pairs of a given instance in a dataset at idx.
    
    Args:
        data: The dataset to inspect.
        idx: The index of the instance to inspect.
    """
    for k, v in data[idx].items():
        print(f"{k}\n{v}")
    print('*'*50)

@torch.no_grad()
def sample_answers(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    chats:List[str],
    max_new_tokens:int,
    num_samples: int = 10,
    top_p: float = 0.85,
    top_k: int = 10,
    temperature: float = 1.0,
    num_beams: int = 1,
    do_sample:bool = True
) -> str:
    assert tokenizer.padding_side == 'left' 
    encosings:torch.Tensor = tokenizer.batch_encode_plus(
        chats,
        return_tensors='pt',
        padding='longest',
    )

    out_tokens = model.generate(
        input_ids=encosings["input_ids"].cuda(),
        attention_mask=encosings["attention_mask"].cuda(),
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_samples,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=num_beams,
        do_sample = do_sample,
    )

    return tokenizer.batch_decode(out_tokens, skip_special_tokens=True)


class GSM8KParser: 

    @classmethod
    def get_answer_from_gt(cls, answer_text:str) -> Dict[str, str]: 
        lines = answer_text.strip().split('\n')
        
        if "####" not in lines[-1]:
            raise ValueError(f"Ill-formed answer provided: {answer_text}")
        
        answer_str:str = lines[-1].replace("####", '').strip()
        answer_str_digit = answer_str.replace(",",'')

        try: 
            eval(answer_str_digit)
        except Exception as e: 
            raise ValueError(f"Ill-formed answer provided: {answer_str}") from e

        return {
            "answer_str_digit": answer_str_digit
        }

    
    @classmethod
    def get_answer_from_pred(cls, pred_answer_text:str) -> Dict[str, str]:

        # positive lookahead, terminate at the end of the string or the next "####"
        answer_pattern = r"(####.*?)(?=\Z|####)" 
        matches:List[str] | None = re.findall(answer_pattern, pred_answer_text, flags=re.DOTALL)
        if not matches:
            return {"answer_str_digit": INVALID_ANSWER}

        last_match = matches[-1].replace('#','').strip()
        last_match = re.sub(r"(?<!\,)\,(?!\,)", '', last_match)
        
        # forward search to cover all digits after ####
        candidate = ''
        for i, c in enumerate(last_match): 
            
            if i == 0 and c == '-':
                candidate += c 
                continue

            if c in VALID_ANSWER_CHARS:
                try:
                    eval(candidate+c)
                    candidate += c
                except Exception:
                    break 
            else:
                break   

        if not candidate: 
            return {"answer_str_digit": INVALID_ANSWER}
        
        return {"answer_str_digit": candidate}


    @classmethod
    def get_num_hops(cls, answer_text:str) -> Dict[str, int]:    
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
        return {
            "num_hops": len(answer_text.strip().split('\n'))-1
        }


    @classmethod
    def extract_equations(text:str) -> List[str] | None: 
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
        return re.findall(pattern, text)


class GMS8KEvaluator: 

    def __init__(self):
        pass     

    
    def _get_maj(self, candidates:List[str]) -> str:
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


    def get_maj_at_k(self, candidates:List[str], answer:str) -> int:
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
        return int(self._get_maj(candidates)  == answer)
    

if __name__ == "__main__":


    ex1 = \
    """
    Together, they made 80+100 = <<80+100=180>>180 pizzas on the second day.
    In the two days, the two of them made a total of 180 +200 = <<180+200=380>>380 pizzas.
    #### 380.00 
    """

    assert GSM8KParser.get_answer_from_gt(ex1) == {'answer_str_digit': '380.00'}
    assert GSM8KParser.get_answer_from_pred(ex1) == {'answer_str_digit': '380.00'}, print(GSM8KParser.get_answer_from_pred(ex1))


    ex2 = \
    """
    # 1+1 = 3 
    # 3 + 2 = 5 
    # so bla bla bla
    #### -789,678,787,878
    """

    assert GSM8KParser.get_answer_from_gt(ex2) == {'answer_str_digit': '-789678787878'}
    assert GSM8KParser.get_answer_from_pred(ex2) == {'answer_str_digit': '-789678787878'}, print(GSM8KParser.get_answer_from_pred(ex2))


    ex3 = \
    """
    # 1+1 = 3 
    # 3 + 2 = 5 
    # so bla bla bla
    #### -78,000,000
    """
    assert GSM8KParser.get_answer_from_gt(ex3) == {'answer_str_digit': '-78000000'}
    assert GSM8KParser.get_answer_from_pred(ex3) == {'answer_str_digit': '-78000000'}, print(GSM8KParser.get_answer_from_pred(ex3))



    ex4 = \
    """
    # 1+1 = 3 
    #### perform calculation 
    #### here is the final answer
    #### -78 
    """ 
    assert GSM8KParser.get_answer_from_gt(ex4) == {'answer_str_digit': '-78'}
    assert GSM8KParser.get_answer_from_pred(ex4) == {'answer_str_digit': '-78'}, print(GSM8KParser.get_answer_from_pred(ex3))
