from textwrap import dedent 

class Template: 

    system = \
    dedent(
    """
    Human:
    You are a helpful assistant that is exceptional at solving Grade School Math Problems. 
    """).strip()

    user = \
    dedent(
    """
    Task: Think step by step to derive the final answer for the following question.
    Approach: Build intermediate steps which leads to the final answer, perform calculations when required.

    ```question
    {question}
    ```
    After completing all steps, you must submit your final answer using only numbers and no other characters.
    Write your final answer in a seperate new line preceeded by "####".
    """).strip()