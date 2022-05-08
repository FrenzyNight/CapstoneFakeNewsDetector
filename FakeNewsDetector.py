import torch
from FNDetectionModel import BERTClassifier
import gluonnlp as nlp
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
import numpy as np
import pandas as pd

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


        col = ['tokenized_idx', 'word', 'weight_acc']
        df = pd.DataFrame(columns = col)
        df_last_idx = 0


        for i, value in enumerate(valid_elems):
            word = self.vocab.idx_to_token[sentence_words_idx[i]]

            if("▁" in word):
                df_last_idx = len(df)
                df.loc[len(df)] = [i, word.replace("▁",""), value]
            elif("[CLS]" in word or "[SEP]" in word):
                continue
            else:
                df.loc[df_last_idx] = \
                [df.loc[df_last_idx]['tokenized_idx'],
                 df.loc[df_last_idx]['word'] + word,
                 df.loc[df_last_idx]['weight_acc'] + value
                ]

        return df

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


        if does_infer_keyword:

            keyword_df = self.infer_keyword(elemwise_final_layer, tokenized, result_idx)

            return keyword_df, result
        else:
            return result



if __name__ == "__main__":

    FNDetector = FakeNewsDetector(use_GPU=True)

    title = "원희룡 “임대차3법, 폐지 가까운 근본적 개선” 무슨 뜻? [뉴스AS]"
    body = """
    '2억 로또' 경기 시흥 아파트 청약에 2만여명이 운집했다. 공공택지에 공급되는 단지로 '분양가 상한제'가 적용돼 주변 시세보다 상대적으로 낮은 가격에 공급된 게 청약 흥행으로 이어졌다. 올해 들어 수도권에서 진행된 청약결과는 '가격'에 따라 희비가 엇갈렸는데, 전문가들은 이 같은 분위기가 이어질 것으로 내다봤다.
    4일 한국부동산원 청약홈에 따르면 전날 1순위 청약을 진행한 'e편한세상 시흥장현 퍼스트베뉴'는 총 67가구 모집에 1만2726명이 몰려 평균 경쟁률 189.94대 1을 기록했다. 가장 높은 경쟁률이 나온 면적대는 전용 84㎡C 기타경기도 624.00대 1을 기록했다. 전용 84㎡A와 전용 84㎡B 기타경기 역시 각각 621.00대1, 623.67대 1로 높은 경쟁률이 나왔다.
    1순위 청약에 앞서 진행된 특별공급에서부터 치열했다. 300가구(기관추천분 제외)를 모집하는 특공에는 7395명이 접수해 평균 경쟁률 24.65대 1이 나왔다. 전용 84㎡A 신혼부부가 34.26대 1로 가장 높은 경쟁률을 기록했고, 전용 84㎡A 생애 최초가 32.30대 1로 뒤를 이었다. 이틀간 진행된 청약에 2만121명이 모인 것이다.
    아파트를 분양받기 위해 예비 청약자들이 몰린 이유는 시세 차익이 기대돼서다. 이 단지 분양가는 최고가 기준 전용면적별로 △84㎡A 4억8486만원 △84㎡B 4억7837만원 △84㎡C 4억6161만원이었다.
    분양 단지 인근에는 지난해 입주한 '시흥장현 동원로얄듀크 2차'와 '시흥장현 금강펜테리움 센트럴파크'가 있지만 아직 정확한 실거래가를 확인하기 어렵고, 맞은 편 능곡동에 있는 '상록힐스테이트'(2009년 입주, 321가구) 전용 84㎡가 지난달 6억5000만원에, '시흥능곡신안인스빌'(2008년 입주, 394가구)이 같은 달 5억8000만원에 손바뀜했다. 적게는 1억6000만원에서 많게는 1억8000만원가량 시세차익이 기대되는 셈이다.
    분양업계 관계자는 "분양가 상한제가 적용돼 주변 시세보다 상대적으로 낮은 가격에 공급된 것이 실수요자를 끌어들였다"고 평가했다.
    올해 들어 수도권에선 가격에 따라 분양 성적이 크게 갈렸다. 올해 서울에서 처음 분양한 강북구 미아동 '북서울자이폴라리스'는 1순위 청약에서 295가구 모집에 1만157명이 신청해 평균 경쟁률 34.40대 1을 기록했다. 하지만 고분양가 논란에 계약 포기자가 나오면서 무순위 청약(줍줍)을 진행해 분양을 마쳤다. 마찬가지 같은 구에서 분양한 '한화포레나 미아'도 높은 분양가로 한 자릿수 경쟁률이 나왔다.
    반면 인천 서구 '힐스테이트검단웰카운티'는 575가구 모집에 4만6070명이 몰려 1순위 평균 경쟁률 80.12대 1을 기록했고, '파주운정디에트르에듀타운'(47.99대 1), '신영지웰운정신도시'(37.26대 1), '제일풍경채검단2차'(30.31대 1) 등도 양호한 성적을 냈다. 이들 단지는 공공택지에 공급된 단지로 주변 시세보다 상대적으로 낮은 가격에 공급돼 청약자들이 주목했다.
    권일 부동산인포 팀장은 "수도권 인구가 지속 증가하고 있는 가운데 가격 경쟁력이 높은 단지 인기는 당분간 지속될 것"으로 내다봤다.
    """

    df, result = FNDetector.inference(title, body)

    print("result:","Fake News" if result else "True news")

    print(df.sort_values(by="weight_acc", ascending=False).head(30))

# 나중에는 그냥 단어를 split한 다음에 집합에 담아놓고 형태소를 contain하고 있으면 weight를 acc하는 방식으로 진행해도 될듯 함