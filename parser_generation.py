from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt
import dmrst
import torch

def pair_completion(model_prompt, relation, model, rst_parser, tokenizer,
                    device = "cuda:0", rst_weight = 0.2, vocality = 0,
                   greedy = True, model_context = None, topk = 100, nucleus = None, banned_ids = None,
                    max_new_tokens = 20):
    '''Given a text prompt and a desired relation, returns text that completes the relation.
    If the prompt is "I would love to come," and the relation is "Contrast_NN", the output
    could be "I would love to come, but I have to work on my homework." Generation ends either when
    a period is encountered, when a new EDU is detected, or when the maximum number of new tokens
    has been reached.
    
    :param model_prompt: the first half of the desired relation that is to be completed
    :param relation: the desired relation that the generated text should fulfil with the
        model prompt
    :param: model: the language model that drives the text generation. The model should be
        a HuggingFace ModelForCausalLM, that it is is callable in a way such that 
        model(tokens).logits[:,-1] returns 1D array of logit values for the next token
    :param rst_parser: should be an instance of DMRST, a wrapper class for the DMRST
        parser
    :param tokenizer: a tokenizer for the language model. Should be a HuggingFace tokenizer
    :param device: the cuda device on which the language model resides
    :param rst_weight: a value from 0 to 1. Higher value means that more power is given
        to the parser to determine the output
    :param vocality: 0 if nothing is to be printed; 1 if the current output sequence should
        be output after each new token generation
    :param greedy: if True, generates the output in a greedy, determinant way; if False,
        samples from a distribution
    :param model_context: extra text prepended to the model input for text generation.
        This text is not used in classifying the relation, ie. the generated text's
        relation to this text is not considered.
        
        For example, if model_prompt = "I love you!" and model_context = "Who's a good dog?",
        the generated text must only hold the realtion with "I love you!" The language model,
        though, will generate text with respect to "Who's a good dog? I love you!"
        
    :param topk: the number of tokens taken from the top of the language model's distribution
        that the rst_parser re-weights
    :param nucleus: floating point number from 0.0 to 1.0.  If None of 1.0, does nothing.
        From the top k tokens taken from the language model's distribution, only as many tokens
        are taken as are required to make the sum the tokens' probabilities >= 'nucleus'. 
        The rst_parser re-weights the tokens left after this parameter filters some less likely
        tokens out. Confer with nucleus sampling from Holtzman et al. (2019) for details.
    :param banned_ids: a list of token ids, integers, which the language model is not allowed
        to generate.
    :param max_new_tokens: the maximum number of tokens that will be generated. If generation
        reached
        
    '''
    
    EOS_TOKEN = "</s>"
    EOS_ID = tokenizer("</s>", return_tensors="pt").input_ids.item()
    
    # Get number of parser tokens in prompt
    len_edu1 = len(rst_parser.bert_tokenizer(model_prompt, add_special_tokens = False).input_ids)
    
    # get # of starter EDUs
    edu_breaks = rst_parser.infer([model_prompt])[1][0]
    num_edus = len(edu_breaks)
    
    if model_context:
        prompt_tokens = tokenizer(model_context + model_prompt,
                                  return_tensors = "pt").input_ids.to(device)
    else:
        prompt_tokens = tokenizer(model_prompt,
                                  return_tensors = "pt").input_ids.to(device)
    generated = ""
    
    # Relation id
    r_id = rst_parser.relation_to_index(relation)
    
    softmax = torch.nn.Softmax(dim = -1)
    
    # Generate text
    gen_tokens = torch.tensor([[]], dtype = torch.int).to(device)
    while len(gen_tokens[0]) < max_new_tokens:
        tokens = torch.cat((prompt_tokens, gen_tokens), dim = -1)
        model_out = softmax(model(tokens).logits[:, -1])
        
        # set the probability of selecting each banned id to 0
        if banned_ids:
            model_out[0, banned_ids] = 0
        
        top = model_out.topk(topk)  
        
        if vocality >= 2:
            ys = [0]
            for y in top.values[0]:
                ys.append(ys[-1] + y.item())
            plt.title(f"Total probability in top {topk}: {torch.sum(top.values, dim = -1).item()}")
            plt.plot(ys)
            plt.show()
        
        # Nucleus sampling (Holtzman et al., 2019)
        if nucleus:
            i = 0
            total = 0
            while i < topk and total < nucleus:
                total += top.values.T[i].item()
                i += 1
            
            if i < topk:
                top = type("", (), {'values':top.values[:, 0:i] / total,'indices':top.indices[:, 0:i]})() 
        
        
        # re-weight tokens using parser
        parser_scores = []
        if rst_weight != 0.0:
            for index in top.indices.T:

                text = model_prompt+generated+tokenizer.decode(index.item())
                total_len = len(rst_parser.bert_tokenizer(text, add_special_tokens= False).input_ids)

                # If whitespace that the BERT tokenizer ignores
                if len_edu1 == total_len:
                    score = 0.0
                else:
                    score = rst_parser.infer(
                        [text],
                        input_EDU_breaks=[[len_edu1 - 1, total_len - 1]]
                    )[-1][0][0][r_id]

                parser_scores.append(score)
        # If rst_weight is 0, using the parser wastes time
        else:
            parser_scores = [1.0 for _ in range(len(top.values))]
            
        temperature = 0.1
        parser_probs = softmax(torch.Tensor(parser_scores).to(device)/temperature)
        
        if vocality >= 2:
            ys = []
            for y in sorted(parser_probs, reverse = True):
                ys.append(y.item())
                
            plt.title(f"Parser Probabilities for top {len(parser_probs)}: {torch.sum(parser_probs, dim = -1).item()}")
            plt.plot(ys)
            plt.show()
            input()
        
        temperature = 0.01
        updated_scores = (
            top.values**(1-rst_weight) *
            parser_probs**(rst_weight)
            /temperature
        )
        
        if greedy:
            next_token = updated_scores.topk(1).indices
        else:
            next_token = torch.multinomial(torch.nn.Softmax(dim = 1)(updated_scores), 1)
        
        gen_tokens = torch.cat((gen_tokens, top.indices[:, next_token.item()][None, :]), dim = -1)
        
        generated = tokenizer.decode(gen_tokens[0])
        
        # If the next EDU has been generated
        edu_breaks = rst_parser.infer([model_prompt + generated])[1][0]
        if len(edu_breaks) > num_edus+1:
            i = 0
            while i < len(edu_breaks) - 1 and edu_breaks[i] <= len_edu1:
                i += 1
            end_of_2 = edu_breaks[i] + 1
            tokens = rst_parser.bert_tokenizer(model_prompt + generated, add_special_tokens = False).input_ids
            generated = rst_parser.bert_tokenizer.decode(tokens[len_edu1:end_of_2])
            break

            
        # If end of sentence
        if (
            "." in tokenizer.decode(top.indices[:, next_token.item()].item())
            ) and len(generated) >= 6:
            break
            
        if vocality >= 1:
            newline = "\n"
            sl = "\\n"
            print(f"\r{generated.replace(newline, sl)}", end = '')
            
    return generated

class RSTGenerator:
    
    def __init__(self, device_lm = "cuda:0", device_parser = "cuda:0"):
        
        self.parser = dmrst.DMRST(device_parser)
        
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7",
                                          cache_dir = "./model-cache/")
        self.model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7",
                                             cache_dir = "./model-cache/").to(device_lm)
        
        self.device_lm = device_lm
        self.device_parser = device_parser
        
    def complete_pair(self, model_prompt, relation, rst_weight = 0.6, vocality = 0,
                 greedy = True, model_context = None, topk = 100, nucleus = None,
                 banned_ids = None, max_new_tokens = 20):
        '''Given a text prompt and a desired relation, returns text that completes the relation.
        If the prompt is "I would love to come," and the relation is "Contrast_NN", the output
        could be "I would love to come, but I have to work on my homework." Generation ends either when
        a period is encountered, when a new EDU is detected, or when the maximum number of new tokens
        has been reached.

        :param model_prompt: the first half of the desired relation that is to be completed
        :param relation: the desired relation that the generated text should fulfil with the
            model prompt
        :param rst_weight: a value from 0 to 1. Higher value means that more power is given
            to the parser to determine the output
        :param vocality: 0 if nothing is to be printed; 1 if the current output sequence should
            be output after each new token generation
        :param greedy: if True, generates the output in a greedy, determinant way; if False,
            samples from a distribution
        :param model_context: extra text prepended to the model input for text generation.
            This text is not used in classifying the relation, ie. the generated text's
            relation to this text is not considered.

            For example, if model_prompt = "I love you!" and model_context = "Who's a good dog?",
            the generated text must only hold the realtion with "I love you!" The language model,
            though, will generate text with respect to "Who's a good dog? I love you!"

        :param topk: The number of tokens taken from the top of the language model's distribution
            that the rst_parser re-weights
        :param nucleus: floating point number from 0.0 to 1.0.  If None of 1.0, does nothing.
            From the top k tokens taken from the language model's distribution, only as many tokens
            are taken as are required to make the sum the tokens' probabilities >= 'nucleus'. 
            The rst_parser re-weights the tokens left after this parameter filters some less likely
            tokens out. Confer with nucleus sampling from Holtzman et al. (2019) for details.
        :param banned_ids: a list of token ids, integers, which the language model is not allowed
            to generate.
        :param max_new_tokens: the maximum number of tokens that will be generated. If generation
            reached

        '''
    
        return pair_completion(model_prompt, relation, self.model, self.parser, self.tokenizer,
                               device = self.device_lm, rst_weight = rst_weight, vocality = vocality,
                               greedy = greedy, model_context = model_context, topk = topk, nucleus = nucleus,
                               banned_ids = banned_ids, max_new_tokens = max_new_tokens)