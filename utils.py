from typing import List 
from transformers import PreTrainedModel, PreTrainedTokenizer


def sample_answers(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    questions: List[str],
    max_length: int = 100,
    num_samples: int = 100,
    top_p: float = 0.9,
    top_k: int = 10,
    temperature: float = 0.2,
    repetition_penalty: float = 1.0,
    num_beams: int = 1,
    ):
    
    in_tokens = tokenizer.batch_decode(questions, return_tensors='pt')

    out_tokens = model.generate(
        input_ids=in_tokens,
        max_length=max_length,
        num_return_sequences=num_samples,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
    )

    return tokenizer.batch_decode(out_tokens, skip_special_tokens=True)