from textwrap import dedent

class EvalTemplateExp:
    # This prompt is used to see if we could elicit the <<>> format as shown from the training set
    # However, after several attempts, it's quite hard to elicit such behaviour from phi-3.5B
    # so we no longer reinforce that inside the prompt

    # In https://arxiv.org/pdf/2308.01825 (Scaling Paper), 
    # the smallest model is llama-7B, I defer to such observation
    # due to the size of the model being much larger in the paper 
    # hence the "instruction-following ability" is better captured by larger models
    # though the ability on math maybe not as strong

    system = dedent(
        """
    You are a highly intelligent assistant who is exceptional at solving Math Problems.
    """
    ).strip()

    user = dedent(
        """
    *Formulas*
    Make sure to annotate calculations with PURE DIGITS in side "<< >>" tags.
    Refer to the example below.
    
    *Answer*
    Write your final answer with PURE DIGITS in a seperate new line preceeded by "####", proceeded by {eos_token}
    Refer to the example below.

    *Example Format*
        User:
        What's the price of 2 eggs, one is $1.50, and the other is $1.00?

        Assistant:
        Let's break down the problem step by step:
        1. We have two eggs, one is $1.50, and the other is $1.00.
        2. The total price of 2 eggs is $1.50 + $1.00 = << 1.50 + 1.00 = 2.50 >> $2.50
        Answer: #### 2.50 <<{eos_token}>>

    *Task*    
    Now think step by step to solve the following question:
    ```question
    {question}
    ```
    """
    ).strip()


class EvalTemplate(EvalTemplateExp):

    user = dedent(
"""
*Task*    
Think step by step to solve the following question:
NOTE:
1. Reason deductively.
1. Write all equations in a single line wihtout breaks in the middle.
2. Submit final answer using PURE DIGITS, in the last starting with "####".
    i.e. "#### 10" if the final answer is $10

```question
{question}
```
"""
).strip()
