# Language Model Sentence Completion with a Parser-Driven Rhetorical Control Method

This repository contains an implementation of the 2024 EACL paper, "Language Model Sentence Completion with a Parser-Driven Rhetorical Control Method."
The paper presents a controlled-text-generation algorithm that enforces adherence toward specific rhetorical relations during LLM sentence-completion.
The code herein is the code used to get the results discussed in the paper.

This repository is made possible by the publically available DMRST parser by Zhengyuan Liu et al. in [their repository](https://github.com/seq-to-mind/DMRST_Parser).
Since the present work depends upon their work, please cite Liu et al. also when citing the present work. The relevent citations for Liu et al. may be found in their [GitHub repository](https://github.com/seq-to-mind/DMRST_Parser).

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
More documentation to come (written February 3rd, 2024).
