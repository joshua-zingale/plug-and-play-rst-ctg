# Language Model Sentence Completion with a Parser-Driven Rhetorical Control Method

This repository contains an implementation of the 2024 EACL paper, "Language Model Sentence Completion with a Parser-Driven Rhetorical Control Method."
The paper presents a controlled-text-generation algorithm that enforces adherence toward specific rhetorical relations during LLM sentence-completion.
The code herein is the code used to get the results discussed in the paper.

This repository is made possible by the publically available DMRST parser by Zhengyuan Liu et al. in [their repository](https://github.com/seq-to-mind/DMRST_Parser).
Since the present work depends upon their work, please cite Liu et al. also when citing the present work. The relevent citations for Liu et al. may be found in their [GitHub repository](https://github.com/seq-to-mind/DMRST_Parser).

## Manifest
- **automatic_evaluation.py** script to run the automatic evaluation results
- **complete_sentences.py** script to generate the sentence completions from the paper
- *dmrst.py* a class written to interface with the Liu et al. DMRST code
- *generaciones* the spanish language prompts and generations evaluated in the paper's appendix
  
More documentation to come (written January 30th, 2024).
