import torch
import string

from transformers import BertTokenizer, BertForMaskedLM

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
top_k = 10


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):

    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    return {'bert': bert,
            }


set = []
sentence1 = 'A person looks straight up, raises his right arm slightly, lowers his left arm, and bends his knees in order to'
sentence2 = 'A person looks straight up, raises his right arm slightly, lowers his left arm, and bends his knees because'

while True:
    response1 = get_all_predictions(sentence1 + ' <mask>', 1)

    if response1['bert'] in set:
        break
    set.append(response1['bert'])
    sentence1 = sentence1 + ' ' + response1['bert']

    if response1['bert'] == '.' or response1['bert'] == '[UNK]':
        break
while True:
    response2 = get_all_predictions(sentence2 + ' <mask>', 1)

    if response2['bert'] in set:
        break
    set.append(response2['bert'])
    sentence2 = sentence2 + ' ' + response2['bert']

    if response2['bert'] == '.' or response2['bert'] == '[UNK]':
        break