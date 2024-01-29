from transformers import AutoModelForCausalLM, AutoTokenizer
import dmrst

# Change this model computation device
# Can be "cpu", "cuda:0", etc.
device = "cpu"

def main():
    
    print("Loading Models, both DMRST and BLOOM 1.7B.",
      "\"Some weights...\" error expected when loading DMRST...")
    parser = dmrst.DMRST(device)

    # model = AutoModelForCausalLM.from_pretrained("gpt2",
    #                             cache_dir = "./model-cache/").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("gpt2",
    #                             cache_dir = "./model-cache/")

    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7",
                                                 cache_dir = "./model-cache/").to(device)
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7",
                                              cache_dir = "./model-cache/")
    
    
    # English generation from human prompts
    with open("generations-from-human-prompts", "r") as f:
        sentences_eh = f.read()

    sentences_eh = sentences_eh.split("""=====
    -----""")[:-1]
    
    bloom_evaluation = evaluate_generations(sentences_eh)
    
    print(bloom_evaluation)
    
    
def get_perplexity(text, model, tokenizer, device = "cuda:0"):
    
    tokens = tokenizer(text, return_tensors = "pt").input_ids.to(device)
    
    t = len(tokens[0]) - 1
    
    logits = model(tokens).logits[0][:-1]
    
    logsoftmax = torch.nn.LogSoftmax(dim = -1)
    log_probs = logsoftmax(logits)
    
    total = torch.sum(log_probs[range(t), tokens[0][1:]])
    
    total /= -t
    
    return torch.exp(torch.Tensor([total])).item()

def parse_relation(relation: str):
    spl = re.split("\(|:|=|,|\)", relation)
        
    _, n_l, r_l, _ = spl[1:5]

    _, n_r, r_r, _ = spl[5:-1]

    rel = r_l
    if rel == "span":
        rel = r_r
    return rel + "_"+n_l[0] + n_r[0]

def evaluate_generations(generations):
    
    pairs = []
    for generation in generations:
        splits = generation.split("\n")

        dic = dict()
        for split in splits:
            if len(split) == 0:
                continue

            key = split[0:split.find(":")]

            if key == "No specified relation":
                key = "None"
            dic[key] = split[split.find(":") + 2:]

        pairs.append(dic)

    out_dict = {
        "num_generations": len(pairs),
        "relations": dict(),
    }
    
    for pair in pairs:

        prompt = pair["Prompt"]
        prompt_num_ids = len(parser.bert_tokenizer(prompt, add_special_tokens = False).input_ids)
        for relation, completion in pair.items():
            if relation == "Prompt":
                continue
            
            if relation not in out_dict["relations"]:
                out_dict["relations"][relation] = {
                    "average_perplexity": 0,
                    "correct": 0,
                    "total": 0,
                    
                }
                
            full_text = prompt + completion

            len_full = len(parser.bert_tokenizer(full_text, add_special_tokens = False).input_ids)
            
            perplexity = get_perplexity(full_text, model, tokenizer, device = "cuda:1")
            out_dict["relations"][relation]["average_perplexity"] += perplexity

            _, _, pred_relation, logits = parser.infer([prompt + completion],
                                                      input_EDU_breaks = [[prompt_num_ids - 1,len_full - 1]])
            
            out_dict["relations"][relation]["total"] += 1
            
            parsed = parse_relation(pred_relation[0][0])
            
            if relation == parsed:
                out_dict["relations"][relation]["correct"] += 1
                
    # Calculate each average with sum / total
    for relation in out_dict["relations"]:
        out_dict["relations"][relation]["average_perplexity"] /= out_dict["relations"][relation]["total"]

    return out_dict

if __name__ == "__main__":
    main()
