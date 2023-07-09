#-*- coding: utf-8 -*-
import json

from sqlalchemy import null

def conll2tagseq(data):
    tokens, preds, args = [],[],[]
    result = []
    for line in data:
        line = line.strip()
        if line.startswith('#'):
            pass
        elif line != '':
            t = line.split('\t')
            token = t[1]
            pred = t[3]
            arg = t[4]
            
            tokens.append(token)
            preds.append(pred)
            args.append(arg)
        else:
            
            sent = []
            sent.append(tokens)
            sent.append(preds)
            sent.append(args)
            
            result.append(sent)
            tokens, preds, args = [],[],[]

    return result
        

def load_srl_data():
    with open('D:\\WorkSpace\\clean_code\\BERT_for_Korean_SRL\\data\\0728_train_conll.conll', encoding='utf-8') as f:
        d = f.readlines()    
    trn = conll2tagseq(d)
    with open('D:\\WorkSpace\\clean_code\\BERT_for_Korean_SRL\\data\\0728_test_conll.conll', encoding='utf-8') as f:
        d = f.readlines()    
    tst = conll2tagseq(d)
    
    return trn, tst


def data2tgt_data(input_data):
    result = []

    for item in input_data:
        ori_tokens, ori_preds, ori_args = item[0],item[1],item[2]

        for idx in range(len(ori_preds)):
            pred = ori_preds[idx]
            if pred != '_':
                if idx == 0:
                    begin = idx
                elif ori_preds[idx-1] == '_':
                    begin = idx
                end = idx
        
        tokens, preds, args = [],[],[]
        
        for idx in range(len(ori_preds)):
            token = ori_tokens[idx]
            pred = ori_preds[idx]
            arg = ori_args[idx]

        
            if idx == begin:
                tokens.append('<tgt>')
                preds.append('_')
                args.append('X')
                
            tokens.append(token)
            preds.append(pred)
            args.append(arg)
            
            if idx == end:
                tokens.append('</tgt>')
                preds.append('_')
                args.append('X')
        sent = []
        sent.append(tokens)
        sent.append(preds)
        sent.append(args)
        result.append(sent)

    return result 



def load_srl_data_for_bert():
    trn_ori, tst_ori = load_srl_data()    
    trn = data2tgt_data(trn_ori)
    tst = data2tgt_data(tst_ori)
    
    return trn, tst

load_srl_data_for_bert()