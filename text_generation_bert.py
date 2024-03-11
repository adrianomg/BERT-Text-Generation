
# This script generates a random text using a BERT model. 
# The script generates a pre-defined total number of words, with a given probability of adding a period before generating every word. 
# The script uses the BERT model to generate the next word in the sentence and transformers library to load the BERT model and the tokenizer.

# Dont take this script as a good example of how to generate text using BERT.
# I am just playing around and trying to understand how to use BERT to generate text.
# BERT is not a traditional language model, but a masked language model. 
# It cannot be used to compute the probability of a sentence like a normal LM, so it is not designed to generate text.

import torch
from transformers import BertTokenizer, BertForMaskedLM

model_name = "bert-large-cased"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

def generate_sentence(input_text, num_candidate_tokens=5, print_candidates = False):
    """
    Generate a sentence by replacing the masked token in the input text with candidate words.

    Args:
        input_text (str): The input text with a masked token.
        num_candidate_tokens (int, optional): The number of candidate tokens (words) to consider. Defaults to 1.

    Returns:
        str: The generated sentence with the masked token replaced by a candidate word.
    """
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)

    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits

    masked_index = mask_token_index[1].item()
    masked_logits = logits[0, masked_index, :]

    masked_probs = torch.softmax(masked_logits, dim=-1)
    top_candidate_words = torch.topk(masked_probs, num_candidate_tokens, dim=-1).indices.tolist()

    # get random value from 0 to num_candidate_words
    random_index = torch.randint(0, num_candidate_tokens, (1,)).item()

    token = top_candidate_words[random_index]

    
    # print the candidate words side by side and highlight the selected word
    if print_candidates:
        print("Candidate tokens:", end=" ")
        for i, word in enumerate(top_candidate_words):
            word_str = tokenizer.decode([word])
            if i == random_index:
                print(f">> {word_str} <<", end=" | ")
            else:
                print(f"{word_str}", end=" | ")
        print()
    

    generated_sentence = ""
    output_ids = input_ids.clone()
    output_ids[0, masked_index] = token
    output_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generated_sentence = generated_sentence + output_sentence

    return generated_sentence


num_words = 50                          # total number of words to generate
period_probability = 0.0                # probability of adding a period
max_period_probability = 3              # maximum probability of adding a period every turn
probability_increase_per_word = 0.5     # increase in probability of adding a period every turn
max_candidate_tokens = 10               # number of generated words each turn to pick from
print_candidates = True                 # print the candidate tokens (words) each turn

input_text = "The quick brown fox ran [MASK]." # initial input text

# generate the first word
generated_sentence = generate_sentence(input_text, num_candidate_tokens=max_candidate_tokens, print_candidates = True)

# generate the next words
for n in range(0, num_words):

    # generate a random number from 0 to 10
    rolled_dice = torch.randint(0, 10, (1,)).item()

    # if the random number is within the to the period probability, then add a period
    # else, increase the period probability by 0.5
    if rolled_dice <= period_probability:
        generated_sentence = generated_sentence[0:-1] + ". [SEP] [MASK]."
        period_probability = 0
    else:
        generated_sentence = generated_sentence[0:-1] + " [MASK]."
        period_probability = period_probability + probability_increase_per_word if period_probability <= max_period_probability else max_period_probability

    # generate the next word
    generated_sentence = generate_sentence(generated_sentence, num_candidate_tokens=max_candidate_tokens, print_candidates = True)

# print the generated text
print(f"Final result: {generated_sentence}") 

