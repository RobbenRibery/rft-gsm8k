# Rejection Sampling Fine-Tuning on GSM-8K
In this document, I provide a high-level summary of this project.
Please refers to Take_Home_Test.ipynb for detials about Hyperparameters, Training Logs, Evaluation Results, etc.
On top of what we asked for, I have put in some additioanl considersation, which I provide a high level overview in this document. Please then head to Take_Home_Test.ipynb for  more details, where I provide a walk through on the assignment

To start with, let's have a look at the
## Results on our take-home assignment:
Through rejection sampling on only 10% of the GSM-8K dataset, I've improved the model performance by 2 ~ 3.5% on Maj@1.
The baseline Maj@1 is measured by using Phi-3.5-mini-insturct, resulting in 82.5% Maj@1 using temp=0.2 sampling at inference. 
My best model, fine-tuned using QLora (4-bit quntization with 16 rank), achieves 86% Maj@1, using the same inference set up.

NOTE:
All evaluation performance is evaluated on 90% of the GSM-8K test dataset (1188) instances, since we've used the other 10% as evalaution set during fine-tuning.
I used the rest of 90% as the hold-out test set, to some extent avoiding my own selection bias. 
(But in a more formal setting giving more time, we still have to evaluate on the remaining 10% of the test set)

## Further Considerations:
In [Yuan et,al](https://arxiv.org/pdf/2308.01825), each question $q_i$ is used to generate a reasoning paths that is used to generate a prediction. However, sampling $k$ reasoning paths for every single question is very computationally intense. The author also mentions that it's very time-consuming. 

Thus I wanted to see if I could achieve improvements by only using the most "useful" questions (i.e. only the top 10% most useful chunck) to perform rejection sampling, making RFT more data efficient.

### Question Selection Creteria 
Perhaps, one way to interpret the usefulness of a question is how many reasoning paths are required to solve it? 
If a question requiring 10 hops (hops = reasoning paths) to be solved at minimum, during the rejection sampling phase, the model would have higher chance to explore more diverse hops (i.e. take round trips, decribe 2 hops using one hop, etc) to solve the question, comparing to questions only requiring 1-3 hops to be solved.

In GSM8K, the nice thing is that each hop/reasoning path is written in a single line. We could use a simple hueristics to estimate the number of hops needed to solve a question:  

```python
utils.GSM8KParser.get_num_hops
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
        return {"num_hops": len(answer_text.strip().split("\n")) - 1} #take away one line to remve the #### final answer
```
In my experiment, I've used the top 747 questions (about 10% of the train set) in terms of number of estimated hops to perform rejection sampling at k=5. 

### Reasoning Path Selection Criteria
In [Yuan et,al](https://arxiv.org/pdf/2308.01825), the Levensthein Distance is used as the utility metric for the reasoning paths. 
Given all correct resoning paths generated by the model, the author select the most "novel" one by maximising its utility, which is the sum of all the pairwise Levenshtein Distance, apart from to itself. Think of it as summing each row inside a 2D matrix by ignoring the diagonal elelements.

But in the paper, the Levenshtein Distance is calculated on the reasoning paths ($r$), which is represented as a string in natural language(to be best of my knolwedge). I want to investigate if we could calculate the utility metric on parsed equations, including its surroding text as well, and still being able to select more diverse reasoning paths. Not to say that's necessarily a better way to do it, but an interersting variation worth exploring.

Please refers to ```utils.GMS8KParse.parse_equations_from_pred``` for more details, as well as a description of the regex pattern applied in such two cases. Overall, we find that using parsed equations without surrounding text is more effective in terms of selecting reasoning paths, which eventually encourges the model to generalise better. 

That said, the regex provided is not perfect. For example is in capaable of handling equaitons span across multiple lines. 

## Project Structure 

- datasets 
    This directory stores data that has passed rejection sampling 
    i.e. 
    *0.5lev-withouttext-0.5leneq-gsm8k_synthetic_data_747instances_5samples*

    means a dataset generate by rejection sampling schema: 

    [-] with 0.5 weight on levenshtein distance and 0.5 weight on equation length when calculating utility

    [-] with the parser not providing text around the equation when parsing

- generations 
This directoy stores token ids that is generated by model during rejection sampling

- results 
.csv files storing model generations, ground turth and maj@1s

## Engineering 

### Cli Usage
#### generation.py 
```bash 
⚡ dev ~/rft-gsm8k python generation.py --help
usage: generation.py [-h] [OPTIONS]

╭─ options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                                                                          │
│ --temperature FLOAT     LLM Sampling Temperature (default: 0.7)                                                                                                  │
│ --num-return-sequences INT                                                                                                                                       │
│                         Number of samples to return for each instance (default: 5)                                                                               │
│ --inference-batch-size INT                                                                                                                                       │
│                         The batch size to use for inference (default: 2)                                                                                         │
│ --reject-sampling-percentage FLOAT                                                                                                                               │
│                         Percentage of the top-tile instances (sorted in terms of hope length)                                                                    │
│                         to participate in rejection sampling (default: 0.1)                                                                                      │
│ --run-name STR          The name of the run, which will be the name of the dataset saved (default: '')                                                           │
│ --generation-path STR   Path of the .pickle file storing the                                                                                                     │
│                         pre-loaded generatoins of the model (default: datasets/corrected-pred-parser-0.5lev-0.5leneq-gsm8k_synthetic_data_747instances_5samples) │
│ --beta-1 FLOAT          Scoring weight on the Levenshtein distance of equations                                                                                  │
│                         The complementary term (1-beta_1) is the weight on the length of the equations (default: 0.5)                                            │
│ --include-text, --no-include-text                                                                                                                                │
│                         Wehther the equations parsed from the generated completions                                                                              │
│                         should include the text (default: False)                                                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

#### lora.py 
```
⚡ dev ~/rft-gsm8k python lora.py --help
usage: lora.py [-h] [OPTIONS]

╭─ options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                                                                         │
│ --evaluate, --no-evaluate                                                                                                                                       │
│                         Whether to perform evaluation or not (default: True)                                                                                    │
│ --train, --no-train     Whether to perform training or not (default: True)                                                                                      │
│ --model-name STR        Name of the model to be loaded (default: microsoft/Phi-3.5-mini-instruct)                                                               │
│ --rft-data-path STR     Path to the RFT dataset, sampled from the base model (default: corrected-pred-parser-1.0lev-gsm8k_synthetic_data_747instances_5samples) │
│ --use-rft-data-only, --no-use-rft-data-only                                                                                                                     │
│                         Whether to use only the RFT data for training or not (default: False)                                                                   │
│ --use-weighted-training, --no-use-weighted-training                                                                                                             │
│                         Whether to use higher weight on the RFT data for training or not (default: False)                                                       │
│ --run-name STR          base name of the run (default: '')                                                                                                      │
│ --seed INT              Mannual Seed for the run (default: 42)                                                                                                  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
