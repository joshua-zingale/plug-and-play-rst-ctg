# Language Model Sentence Completion with a Parser-Driven Rhetorical Control Method

This repository contains the implementation of our 2024 EACL paper, [Language Model Sentence Completion with a Parser-Driven Rhetorical Control Method](https://aclanthology.org/2024.eacl-short.18/) (Zingale and Kalita, 2024).
The paper presents a controlled-text-generation algorithm that enforces adherence toward specific rhetorical relations during LLM sentence-completion.
The code herein is the code used to get the results discussed in the paper.

This repository is made possible by the publically available DMRST parser by Zhengyuan Liu et al. in [their repository](https://github.com/seq-to-mind/DMRST_Parser).
Since the present work depends upon their work, please cite Liu et al. also when citing the present work. The relevent citations for Liu et al. may be found in their [GitHub repository](https://github.com/seq-to-mind/DMRST_Parser).

If anything in this repository helps you with a publication, please cite our paper according to the following BibTeX:
```
@inproceedings{zingale-kalita-2024-language,
    title = "Language Model Sentence Completion with a Parser-Driven Rhetorical Control Method",
    author = "Zingale, Joshua  and
      Kalita, Jugal",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-short.18",
    pages = "193--203",
    abstract = "Controlled text generation (CTG) seeks to guide large language model (LLM) output, that statistical language generation would conform to desired criteria. The current study presents a novel CTG algorithm that enforces adherence toward specific rhetorical relations in an LLM sentence-completion context by a parser-driven decoding scheme that requires no model fine-tuning. The method is validated both with automatic and human evaluation.",
}
```

## Manifest
- **automatic_evaluation.py** script to run the automatic evaluation results
- **complete_sentences.py** script to generate the sentence completions evaluated the paper
- *dmrst.py* a class written to interface with the Liu et al. DMRST code
- *generaciones* the spanish language prompts and generations evaluated in the paper's appendix
- *generations-from-human-prompts* the English language prompts and generations evaulated in the paper
- *gpt-spanish-prompts* the Spanish language prompts used in the paper
- *human-english-prompts* the English language prompts used in the paper
- **human_evaluation.py** script to process the human evaluation scores and to generate the paper's table values
- *parser_generation.py* a class that implements the main algorithm discussed in the paper
- *requirements.txt* the Python package requirements
- *DMRST_Parser_main* the folder containing a modified version of the Liu et al. DMRST code
- *human-eval* the folder containing the human evaluation scores for each of the three evaluators

The bolded files are those Python scripts that can be run to reproduce the generations, the automatic evaluation of the generations, and the analysis of the human-evaluation scores used in the paper.


## Setup
After cloning this repository to a computer, you must install the required libraries.
This can be done with

```bash
pip install -r requirements.txt
```

Now, the model weights for DMRST need to be downloaded from a link in its [repository](https://github.com/seq-to-mind/DMRST_Parser), which can be found in **depth_mode/Savings/README.md**. After downlowding **multi_all_checkpoint.torchsave**, place it inside the cloned repository with the same name at **DMRST_Parser_main/depth_mode/Savings/multi_all_checkpoint.torchsave**.

The first time BLOOM, the language model, is called, it will be downloaded and cached into this repository's file structure.
This happens automatically when you run one of the scripts.

### GPU
By default, we have configured everything to run via CPU, which will be very slow for anything involving the two models.
Each of the script files, described below, can be configurd to run the language model and DMRST on GPU by changing the variables' values at the top of the script.

## Generation Completions from Paper
Running **complete_sentences.py** will read all of the prompts, first English then Spanish, used in the paper and generate the completions for them.
We used greedy generations so you should get the same results as were used for the paper's statistics, i.e. they should match **generations-from-human-prompts** and **generaciones** (the Spanish generations). The prompts used by this script are read from **human-english-prompts** and **gpt-spanish-prompts**.

At the top of the script, DEVICE_1 can be set to a different value to enable a GPU for the language model and DEVICE_2 can be set to enable A GPU for DMRST.

The generations will be printed to the standard output.

## Automatic Evaluation
Running **automatic_evaluation.py** will read the generations used in the paper, found in **generations-from-human-prompts** and **generaciones**, and then use DMRST to evaluate the relationship between each prompt-generation pair and use BLOOM 1.7B to calculate the perplexity for the generations. This will take a long time, perhaps more than a day, if a GPU is not used.
The results will be printed to the standard output as a dictionary.

To configure this script to use a GPU, change the variable at the top of the script, *device*, to something like "cuda:0".


## Human Evaluation
Running **human_evaluation.py** will read in all of the scores by the human annotators, which are found in **human_eval/**, and then return the averages used in the paper.
