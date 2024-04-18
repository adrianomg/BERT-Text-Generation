# This script tries to generate random text using a BERT model. 
# The script generates a pre-defined number of words, with a given probability of adding a period before generating every word. 
# The script uses the BERT model to predict the next word in the sentence and the transformers library to load the BERT model and the tokenizer.

# Don't take this script as a good example of how to generate text using BERT.
# I am just playing around and trying to understand how to use BERT to generate text.
# This implementation has many limitations, such as not allowing numbers, special characters, or punctuation in the generated text.
# I feed the model just with the words and the [MASK] token so it does not get into a loop of generating random numbers and special characters with no sense.

# Be aware that BERT is not a traditional language model but a masked one. 
# It cannot be used to compute the probability of a sentence like a normal LM, so it is not designed to generate text.

# That said, enjoy!

import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

from time import time

model_name = "bert-large-cased"

config = BertConfig.from_pretrained(model_name, output_hidden_states=True, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.2)

tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertForMaskedLM.from_pretrained(model_name, config=config)

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


    # get a random value from 0 to num_candidate_words
    random_index = torch.randint(0, num_candidate_tokens, (1,)).item()
    token = top_candidate_words[random_index]

    #if token is not a word and it is not different from the previous token, get a new token from the beginning of the list and keep trying until the end
    if len(input_text.split(" ")) > 3:
        previous_token = input_text.split(" ")[-2] if input_text.split(" ")[-1] != "[SEP]" else input_text.split(" ")[-3]
        iterations = 0
        while not tokenizer.decode([token]).isalpha() or tokenizer.decode([token]).lower() == previous_token.lower():
            next_index = (random_index + 1) % num_candidate_tokens
            token = top_candidate_words[next_index]
            iterations += 1
            if iterations == num_candidate_tokens:
                break

    # If none of the candidate words is a valid word, just take a random word from the input text
    if not tokenizer.decode([token]).isalpha():
        token = torch.randint(0, len(tokenizer.vocab), (1,)).item()
    
    # print the candidate words side by side and highlight the selected word
    if print_candidates:
        print("Candidate tokens:", end=" ")
        for i, word in enumerate(top_candidate_words):
            word_str = tokenizer.decode([word])
            print(f"{word_str}", end=" | ")

        print(" -> Selected token:", tokenizer.decode([token]), end="\n")
       

    generated_sentence = ""
    output_ids = input_ids.clone()
    output_ids[0, masked_index] = token
    output_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generated_sentence = generated_sentence + output_sentence

    return generated_sentence


num_words = 50                          # total number of words to generate

max_period_probability = 0.2            # maximum probability of adding a period every turn
probability_increase_per_word = 0.03    # increase in the probability of adding a period every turn
start_period_probability = probability_increase_per_word # probability of adding a period

max_candidate_tokens = 5                # number of generated words each turn to pick from

print_candidate_tokens = False          # print the candidate tokens (words) each turn

input_text = "The quick brown fox ran." # initial input text

start_t = time()

# generate the first word
#generated_sentence = generate_sentence(input_text, num_candidate_tokens=max_candidate_tokens, print_candidates = True)

generated_sentence = input_text

# generate the next words
for n in range(0, num_words):

    # generate a random number from 0 to 10
    rolled_dice = torch.randint(0, 10, (1,)).item()

    # If the random number is within the period probability, then add a period
    # Else, increase the period probability by 0.5
    if rolled_dice <= start_period_probability:
        generated_sentence = generated_sentence[0:-1] + ". [SEP] [MASK]."
        start_period_probability = 0
    else:
        generated_sentence = generated_sentence[0:-1] + " [MASK]."
        start_period_probability = start_period_probability + probability_increase_per_word if start_period_probability <= max_period_probability*10 else max_period_probability

    # generate the next word
    generated_sentence = generate_sentence(generated_sentence, num_candidate_tokens=max_candidate_tokens, print_candidates = print_candidate_tokens)

end_t = time()

# print the generated text
print(f"\n     Generated text: \n{generated_sentence}") 
print("\n     Inference time:", end_t - start_t, end="\n")
