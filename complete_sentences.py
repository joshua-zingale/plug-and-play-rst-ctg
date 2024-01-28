from transformers import AutoTokenizer
import parser_generation
import numpy as np
import dmrst
import torch
import json
import gc

# "cpu" or "cuda:0", "cuda:1"...
DEVICE_1 = "cpu"
DEVICE_2 = "cpu"

# Load and format human generated English sentences
with open("human-english-prompts", 'r') as f:
    sentences1 = f.read()

sentences1 = sentences1.split('\n')
for i in range(len(sentences1)):
    sentences1[i] = sentences1[i] + ", "
print(f"Loaded {len(sentences1)} English sentences")
    

# Load and format ChatGPT generated Spanish sentences
with open("gpt-spanish-prompts", 'r') as f:
    sentences2 = f.read()

sentences2 = sentences2.split("\n")
for i in range(len(sentences2)):
    sentences2[i] = sentences2[i][:sentences2[i].find(".")] + ", "
print(f"Loaded {len(sentences2)} Spanish sentences")

# Which relations will be used for the generation tests
USED_RELATIONS = [
    "Elaboration_NS",
    "Contrast_NN",
    "Cause_NS",
    "Manner-Means_NS",
    "Evaluation_NS",
    "Condition_NS",
    "Joint_NN",]


# Ban any tokens containing at least one of three spacing characters
# (\n, \t, \r) from being generated

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7",
                                          cache_dir = "./model-cache/")
banned_ids = []
for token_id in range(len(tokenizer)):
    for banned in ("\n","\t","\r"):
        if banned in tokenizer.decode(token_id):
            banned_ids.append(token_id)

# Get text generation pipeline
print("Loading Models, both DMRST and BLOOM 1.7B.",
      "\"Some weights...\" error expected when loading DMRST...")
g = parser_generation.RSTGenerator(DEVICE_1, DEVICE_2)



def test_prompt(prompt):
    '''Test a prompt both with no relation being inforced and
    with each of the seven USED_RELATIONS'''
    print(f"Prompt: {prompt}")

    o = g.complete_pair(prompt, 'Contrast_NN', rst_weight = 0.0, vocality = 0, topk = 100,
                        nucleus = 0.75, banned_ids = banned_ids, max_new_tokens = 30)

    print(f"No specified relation: {o}")
    for relation in USED_RELATIONS:
        o = g.complete_pair(prompt, relation, rst_weight = 0.7, vocality = 0, topk = 100,
                            nucleus = 0.75, banned_ids = banned_ids, max_new_tokens = 30)
        print(f"{relation}: {o}")
        
# Generate sentence completions
print("---ENGLISH SENTENCE COMPLETION---")
for sentence in sentences1:
    test_prompt(sentence)
    gc.collect()
    torch.cuda.empty_cache()
    print("=====\n-----")
    
print("---SPANISH SENTENCE COMPLETION---")
for sentence in sentences2:
    test_prompt(sentence)
    gc.collect()
    torch.cuda.empty_cache()
    print("=====\n-----")
