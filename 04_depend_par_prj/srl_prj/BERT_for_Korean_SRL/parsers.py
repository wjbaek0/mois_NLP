import json
import sys

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import os
import bert_srl
import preprocessor
import dataio

sys.path.insert(0,'../')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class srl_parser():
    def __init__(self, model_dir = '/model/bert_ko_srl_model.pt', batch_size=1):
        
        try:
            self.dir_path = os.path.dirname(os.path.abspath( __file__ ))
        except:
            self.dir_path = '.'

        try:
            self.model_dir = model_dir
        except:
            self.model_dir = '.\\model\\bert_ko_srl_model.pt'
        
        model_dir = os.path.dirname(os.path.abspath( __file__ )) +'\\model\\bert_ko_srl_model.pt'

        try:

            self.model = torch.load(self.model_dir)
            self.model.to(device)
            self.model.eval()
        except KeyboardInterrupt:
            raise
        except:
            print('model dir', self.model_dir, 'is not valid ')
            
        self.bert_io = bert_srl.for_BERT(mode='test')
        self.batch_size = batch_size
        
    def ko_srl_parser(self, text):
        
        input_data = preprocessor.preprocessing(text)        
        input_tgt_data = preprocessor.data2tgt_data(input_data)    
        # input_tgt_data = dataio.data2tgt_data(input_data)      
        input_data_bert = self.bert_io.convert_to_bert_input(input_tgt_data) 
        input_dataloader = DataLoader(input_data_bert, batch_size=self.batch_size)
    
        pred_args = []
        for batch in input_dataloader:

            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_orig_tok_to_maps, b_input_masks = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_masks)
            logits = logits.detach().cpu().numpy()
            b_pred_args = [list(p) for p in np.argmax(logits, axis=2)]
            
            for b_idx in range(len(b_pred_args)):
                
                input_id = b_input_ids[b_idx]
                orig_tok_to_map = b_input_orig_tok_to_maps[b_idx]                
                pred_arg_bert = b_pred_args[b_idx]

                pred_arg = []
                for tok_idx in orig_tok_to_map:
                    if tok_idx != -1:
                        tok_id = int(input_id[tok_idx])
                        if tok_id == 1:
                            pass
                        elif tok_id == 2:
                            pass
                        else:
                            pred_arg.append(pred_arg_bert[tok_idx])    

                print("")
                pred_args.append(pred_arg)
                
        pred_arg_tags_old = [[self.bert_io.idx2tag[p_i] for p_i in p] for p in pred_args]

        result = []
        for b_idx in range(len(pred_arg_tags_old)):
            pred_arg_tag_old = pred_arg_tags_old[b_idx]
            pred_arg_tag = []
            for t in pred_arg_tag_old:
                if t == 'X':
                    new_t = 'O'
                else:
                    new_t = t
                pred_arg_tag.append(new_t)
                
            instance = []
            instance.append(input_data[b_idx][0])
            instance.append(input_data[b_idx][1])
            instance.append(pred_arg_tag)
            
            result.append(instance)
        
        return result
            


if __name__ == "__main__" :
    try:
        dir_path = os.getcwd()
    except:
        dir_path = '.'

    p = srl_parser(model_dir = dir_path+'\\model\\ko-srl_0726-epoch-7.pt')

    d = p.ko_srl_parser(text="숭례문에서 방화로 인한 화재가 발생하여 12층에 피해가 발생하고 진화되었습니다.")

    print(d)
