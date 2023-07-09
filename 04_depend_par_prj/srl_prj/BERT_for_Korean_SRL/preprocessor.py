import json
import sys
sys.path.insert(0,'../')
import jpype
from konlpy.tag import Kkma
kkma = Kkma()

# kkma 형태소 분석 
def pred_identifier(word):
    jpype.attachThreadToJVM()
    morps = kkma.pos(word)
    v = False
    result = []
    for m,p in morps:
        if p == 'XSV' or p == 'VV':
            v = True

    if v:
        for i in range(len(morps)):
            m,p = morps[i]
            if p == 'VA' or p == 'VV':
                if m[0] == word[0] and len(m) >= 1:
                    result.append(m)
                    break
            if i > 0 and p == 'XSV' :  
                r = morps[i-1][0]+m
                if r[0] == word[0]:
                    result.append(r)
     
    return result


# 띄어쓰기 단위로 토큰화 
def basic_tokenizer(text):
    tokens = text.split(' ')
    idxs = []
    for i in range(len(tokens)):
        idxs.append(str(i))
    return idxs, tokens


# 서술어부분 앞 뒤로 <tgt> </tgt> 태깅 
def data2tgt_data(input_data):
    result = []
    for item in input_data:
        ori_tokens, ori_preds = item[0],item[1]
        for idx in range(len(ori_preds)):
            pred = ori_preds[idx]
            if pred != '_':
                if idx == 0:
                    begin = idx
                elif ori_preds[idx-1] == '_':
                    begin = idx
                end = idx

        tokens, preds = [],[]
        for idx in range(len(ori_preds)):
            token = ori_tokens[idx]
            pred = ori_preds[idx]
            if idx == begin:
                tokens.append('<tgt>')
                preds.append('_')

            tokens.append(token)
            preds.append(pred)

            if idx == end:
                tokens.append('</tgt>')
                preds.append('_')
        sent = []
        sent.append(tokens)
        sent.append(preds)
        result.append(sent)
    return result 


def preprocessing(text):
    result = []

    idxs, tokens = basic_tokenizer(text)

    for idx in range(len(tokens)):
        token = tokens[idx]
        verb_check = pred_identifier(token)
        
        if verb_check:
            preds = ['_' for i in range(len(tokens))]
            preds[idx] = verb_check[0]+'.01'
            instance = []            
#             instance.append(idxs)
            instance.append(tokens)
            instance.append(preds)
            result.append(instance)
            
    return result