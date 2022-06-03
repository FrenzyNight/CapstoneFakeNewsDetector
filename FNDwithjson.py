#from sqlalchemy import false
import torch
from FNDetectionModel import BERTClassifier
import gluonnlp as nlp
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
import numpy as np
import json

import sys

class FakeNewsDetector:
    def __init__(self, max_len = 256,use_GPU = False):

        self.max_len = max_len
        self.use_GPU = use_GPU

        if(use_GPU and torch.cuda.is_available()):
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.bertmodel, self.vocab = get_pytorch_kobert_model(cachedir=".cache")
        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)

        self.model = BERTClassifier(self.bertmodel,  dr_rate=0.5).to(self.device)
        self.model.load_state_dict(torch.load('Parameter/FNDetectorParameters.pt', map_location = self.device)) # state_dict를 불러 온 후, 모델에 저장

    def infer_keyword(self, elemwise_final_layer, tokenized, result_idx):
        elemwise_final_layer = elemwise_final_layer.detach().numpy()[result_idx]
        elem_mu = np.mean(elemwise_final_layer)
        elem_sigma = np.std(elemwise_final_layer)

        baseline = elem_mu + elem_sigma

        valid_elems = [i if i>baseline else 0 for i in elemwise_final_layer]
        valid_elems = valid_elems[:self.max_len]

        sentence_words_idx = tokenized[0]


        word_weight_list = []
        last_idx = 0

        for i, value in enumerate(valid_elems):
            word = self.vocab.idx_to_token[sentence_words_idx[i]]

            if("▁" in word):
                #word.replace("▁","")
                last_idx = len(word_weight_list)
                word_weight = {
                    'word' : word.replace("▁",""),
                    'weight' : value
                }

                word_weight_list.append(word_weight)
            elif("[CLS]" in word or "[SEP]" in word):
                continue
            else:
                word_weight_list[last_idx]['word'] = word_weight_list[last_idx]['word'] + word
                word_weight_list[last_idx]['weight'] = word_weight_list[last_idx]['weight'] + value
                

        return word_weight_list

    def inference(self, title, body, does_infer_keyword = True):

        transform = nlp.data.BERTSentenceTransform(self.tok, max_seq_length=self.max_len, pad=True, pair=True)
        tokenized = transform((title, body))

        token_ids, valid_length, segment_ids = torch.from_numpy(tokenized[0]) \
                , torch.from_numpy(tokenized[1]) \
                , torch.from_numpy(tokenized[2]) 



        token_ids = token_ids.unsqueeze(0).long().to(self.device)
        valid_length = valid_length.unsqueeze(0).long().to(self.device)
        segment_ids = segment_ids.unsqueeze(0).long().to(self.device)

        elemwise_final_layer, out = self.model(token_ids, valid_length, segment_ids)


        _, result_idx = torch.max(out,1)
        result_idx = result_idx.cpu().numpy()[0]

        result = True if result_idx==1 else False

        sigmoid = torch.nn.Sigmoid()
        per = sigmoid(out)
        if(per[0][0] >  per[0][1]):
            val = per[0][0].item()
        else:
            val = per[0][1].item()

        score = (int)(val*100)

        if does_infer_keyword:

            word_weight_list = self.infer_keyword(elemwise_final_layer, tokenized, result_idx)

            return word_weight_list, result, score
        else:
            return result


def read_arguments():

    l = len(sys.argv)
    if(l != 4):
        raise Exception("There must be three of arguments \
        \"python FNdwithjson.py input.json output.json words_len\""+
        "\n ther are "+str(l) + " of arguments.")

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    try:
        words_len = int(sys.argv[3])
    except:
        raise Exception("Third argument must be a integer.")

    return input_file_name, output_file_name, words_len 

def read_input_from_json(input_file_name):

    with open(input_file_name, 'r', encoding='utf-8') as f:
        f = json.load(f)
        title = f['title']
        body = f['body']


    return title, body
    

if __name__ == "__main__":


    input_file_name, output_file_name, words_len = read_arguments()
    title, body = read_input_from_json(input_file_name)

    
    FNDetector = FakeNewsDetector(use_GPU=True)

    word_weight_list, result, score = FNDetector.inference(title, body)

    word_weight_list = sorted(word_weight_list, key=(lambda x: x['weight']), reverse=True)
   
    word_weight_list = word_weight_list[:words_len]
    



    new_data = {
    'TF' : result,
    'Score' : score,
    'Keywords' : word_weight_list
    }



    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False)

    print("result:","Fake News" if result else "True news")

