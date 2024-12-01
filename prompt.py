from textwrap import dedent

class EvalTemplateExp:
    # This prompt is used to see if we could elicit the <<>> format as shown from the training set
    # However, after several attempts, it's quite hard to elicit such behaviour from phi-3.5B
    # so we no longer reinforce that inside the prompt 

    # In https://arxiv.org/abs/2110.14168, the smallest model is 7B, I defer to such observation 
    # due to the size of the model

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
    ## We are not inserting on the output format anymore, so we better be smart with 
    ## ways that we can compare reasoning reaces later on 
    user = dedent(
    """
*Task*    
Think step by step to solve the following question:
```question
{question}
```

*Format*
Submit the final answer with PURE DIGITS in a seperate new line starting with "####", proceeded by <|end|>
i.e. if the answer is £10, then write:"\n#### 10 <|end|>"
    """
    ).strip()



class RejectSamplingTemplate(EvalTemplateExp):

    user = dedent(
    """
    \n*Task 
    You will solve a math question requiring deductive reasoning.

    *Format
        **Formulas
        Make sure to perform any calculations with PURE DIGITS in side "<< >>" tags, and quote the result outside the tag once complete.
        i.e. 1 (egg) + 1 (egg) = <<1 + 1 = 2>> 2 eggs. Noticed that formula inside the << >> tags does not contain units.

        **Final Answer
        Write your final answer with PURE DIGITS in a seperate new line preceeded by "####", proceeded by {eos_token}
        i.e. #### 2 {eos_token}
    
    *Example
    User:
    What's the price of 2 eggs, one is $1.50, and the other is $1.00?

    Assistant:
    Let's break down the problem step by step:
    1. We have two eggs, one is $1.50, and the other is $1.00.
    2. The total price of 2 eggs is $1.50 + $1.00 = << 1.50 + 1.00 = 2.50 >> $2.50
    #### 2.50 <<{eos_token}>>

    Now think step by step to solve the following question, be as detailed as possible.
    ```question
    {question}
    ```
    """
    ).strip()
