"""
# 파일명 : 
# 설명   : 
# 개정이력 :
#    버젼    수정일자                   수정자              내용
#    1.0    2022-05-31                  김종완            신규 작성
""" 
import jsonlines
import os
import glob
import pandas as pd
import numpy as np
import re
import argparse
import traceback
import datetime
import time
import json
import uuid
from tqdm import tqdm  # processbar
from typing import Tuple
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer

# common
from common import DefineEntitiesName
from common import Params
from common import Paths
from common import RegExpObject
from common import createFolder

# processor
from processor import TrainParams
from processor import AbstractPreprocessor
from processor import DataParsing

#logging setting
from common import Logging_config
logging_instance = Logging_config(Params.CUSTOM_LOG_FILE, Paths.LOG_DIR)
logger = logging_instance.logging_setting()



class CausalDataPreprocessor(AbstractPreprocessor):
    DATASET_NAME      = "CausalData" # FIXME 요 데이터 이름 변경 해야함
    NO_RELATION_LABEL = "Other"      # FIXME 이 other도 없애야 하지 않을까??

    def __init__(self, tokenizer: PreTrainedTokenizer, train_data_path):
        super().__init__(tokenizer) # TODO 여기서 AbstractPreprocessor init으로 넘어감
        self.read_train = self.read_tsv(train_data_path)
    

    def read_tsv(self, data_path):
        raw_tsv_df = pd.read_csv(data_path, sep="\t",encoding="utf-8", header=None, names=DataParsing.TSV_NAMES) # DataParsing.TSV_NAMES : list
        return raw_tsv_df
                                      

    # TODO 3.
    def _preprocess_data(self):
        # train test split
        split_train_df, split_test_df = train_test_split(self.read_train, shuffle=True, random_state=42, test_size=0.1)
        logger.info("Processing training data")
        train_data = self._process_file(split_train_df)

        logger.info("Processing test data")
        test_data = self._process_file(split_test_df)

        logger.info("Splitting train & validate data")
        train_data, val_data = train_test_split(train_data, shuffle=True, random_state=42, test_size=0.2)

        return train_data, val_data, test_data

    # def _preprocess_data(self):
    #     # train test split
    #     split_train_df, split_test_df = train_test_split(self.read_train, shuffle=True, random_state=42, test_size=0.2)
    #     train_data, val_data          = train_test_split(split_train_df, shuffle=True, random_state=42, test_size=0.1)

    #     # 이쪽에 csv 저장 로직

    #     train_data = self._process_file(train_data)
    #     val_data   = self._process_file(val_data)
    #     test_data  = self._process_file(split_test_df)


    #     return train_data, val_data, test_data
    

    def _process_file(self, file_name) -> DataFrame:
        raw_sentences = []
        labels = []

        for _, val in file_name.iterrows() : 
            sent = val[DataParsing.TSV_NAMES[3]] # "sentence"
            label, sub, obj = self._process_label(val)
            labels.append(label)             # TODO label append
            raw_sentences.append(self._process_sentence(sent, sub, obj))

        return self._clean_data(raw_sentences, labels) # MEMO data 정합성 검사 _clean_data

    
    @staticmethod
    def _process_sentence(sentence: str, sub: int, obj: int) -> str:
        # MEMO  여기서 텝으로 구분하고 뒤에 개행 빼고 [ or { 로 바꿈
        # 0 : start , 1 : end
        sentence_replace = sentence \
            .replace(f"<e{sub}>", TrainParams.SUB_OBJ_DICT['SUB_CHAR'][0]) \
            .replace(f"</e{sub}>",TrainParams.SUB_OBJ_DICT['SUB_CHAR'][1]) \
            .replace(f"<e{obj}>", TrainParams.SUB_OBJ_DICT['OBJ_CHAR'][0]) \
            .replace(f"</e{obj}>",TrainParams.SUB_OBJ_DICT['OBJ_CHAR'][1])

        return sentence_replace


    @staticmethod
    def _process_label(val) -> Tuple[str, int, int]:
        label = val[DataParsing.TSV_NAMES[0]]                  # "relation"
        find_token_seq = DataParsing.search_token(val)         # search token해서 있는 문장만 찾기
        nums = list(filter(str.isdigit, str(find_token_seq))) # 원인과 결과 번호 구분 # TODO 이거 왜 이렇게 구분으로 하는지...

        return label, int(nums[0]), int(nums[2])

   
    @classmethod
    def main(cls, tsv_save_path):
        pretrained_model = TrainParams.AVAILABLE_PRETRAINED_MODELS[5] # kcbert
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        # some tokenizer, like GPTTokenizer, doesn't have pad_token
        # in this case, we use eos token as pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # processor 실행
        preprocessor = CausalDataPreprocessor(tokenizer       = tokenizer,
                                              train_data_path = tsv_save_path,
                                             )
        preprocessor.preprocess_data()

        return None



"""
# Class  : PreprocessJsonData
# 설명   : jsonl 데이터 token 추가 및 tsv형식으로 전처리
"""
class PreprocessJsonData(DefineEntitiesName):
    def __init__(self):
        self.process_ele_dict      = self.INIT_DICT                      # common의 초기 빈값의 dictionary
        self.token_key_list        = list(self.CAUSAL_TOKEN_DICT.keys()) # ['원인', '결과']
        self.integrate_key_list    = list(self.INTEGRATED_DICT.keys())   # 통합 format json key list
        self.redifine_position_val = Params.REDIFINE_POSITION_VAL        # NIA 표준 json 생성시 조정할 위치값
        return
    

    """
    # 함수명  : read_jsonl
    # 설명    : jsonl파일을 읽어온다.
    # return  : jsonl_element_list
    # 특이사항 : jsonl 파일 읽어서 리스트 반환
    """
    def read_jsonl(self, jsonl_path):
        jsonl_element_list = []  # jsonl 요소담을 빈 list
        # read jsonl 
        with jsonlines.open(jsonl_path) as f :
            for line in f:        
                jsonl_element_list.append(line)

        logger.info(f"======== Labeling Text Data Count : {len(jsonl_element_list)}")

        return jsonl_element_list


    """
    # 함수명  : create_str_sentence
    # 설명    : 인과관계 토큰이 있는 문장만 만든다.
    # return  : sentence_str
    # 특이사항 : 정규식 사용
    """
    def create_str_sentence(self, ele_text_list):
        # TODO 문장은 이제 무조건 한문장이 되고 문장 나눔은 개행문자("\n")으로 바꾸는 작업
        sentence_list = "".join(ele_text_list).split(".")
        # sentence_list = "".join(ele_text_list).split("\n")

        regex = re.compile(RegExpObject.SEARCH_TOKEN) # 인과관계 토큰 정규식
        
        for sentence in sentence_list :
            if regex.search(sentence) :
                sentence_str = f'{sentence.strip()}.' # dot 추가
                # sentence_str = f'{sentence_list}' # dot 추가
                pass  # 토큰이 있는 문장을 찾았으니 pass

        return sentence_str


    """
    # 함수명  : __search_entity
    # 설명    : 원인과 결과 entity를 dictionary에 추가
    # return  : None
    # 특이사항 : 
    """
    def __search_entity(self, copy_text_list, entities_list):
        for causal_dict in entities_list : 
            causal_str = "".join(copy_text_list[causal_dict["start_offset"] : causal_dict["end_offset"]]).strip() # 원인 또는 결과 entity 생성
            temp_key   = self.CAUSAL_KEY_DICT[causal_dict['label']] # dictionary 리스트에 넣기 위해 key찾아줌 caus or result
           
            # entity append
            self.process_ele_dict[temp_key].append(causal_str)

        return None


    """
    # 함수명  : __insert_token
    # 설명    : entities 추가 및 토큰 위치 조정하여 추가
    # return  : copy_text_list
    # 특이사항 : 토큰 추가시 위치에 따른 index추가 조정이 된다.
    """
    def __insert_token(self, copy_text_list, entities_list):
        # 원인과 결과 entity string append
        self.__search_entity(copy_text_list, entities_list)

        start_offset_list = [ele['start_offset'] for ele in entities_list] # start_offset list

        # 위치를 조정하여 토큰 추가
        add_index = 0 # 토큰 추가시 밀려나는 index 초기값
        for no, (offset, casual_dict) in enumerate(zip(start_offset_list, entities_list)) : 
            # 첫번째가 아닌데 start_offset이 이전에 것 보다 클때
            if (no > 0) and (offset > start_offset_list[0]) :
                add_index = 2
            
            # 원인 토큰 insert e1 or e2
            copy_text_list.insert(casual_dict['start_offset'] + add_index,     f"<{self.CAUSAL_TOKEN_DICT[casual_dict['label']]}>")
            copy_text_list.insert(casual_dict['end_offset']   + add_index + 1, f"</{self.CAUSAL_TOKEN_DICT[casual_dict['label']]}>")

        return copy_text_list


    """
    # 함수명  : insert_causal_token_df
    # 설명    : 원인과 결과에 토큰을 추가하고 tsv를 만들기 위한 dictionary를 만든다.
    # return  : None
    # 특이사항 : DataFrame화 후 해당 entity정보 찾아서 다시 dictionary화
    """
    def insert_causal_token_df(self, jsonl_element_list):
        pbar = tqdm(total=len(jsonl_element_list)) # processbar setting

        self.INTEGRATED_DATA_DICT['create_date'] = str(datetime.datetime.now()) # create time in dictionary

        for element in jsonl_element_list : 
            ele_text_list = list(element['text'])              # 리스트화해서 한글자마다 인덱싱
            relations_df  = pd.DataFrame(element['relations']) # relations를 dataframe으로 만든다.
            entities_df   = pd.DataFrame(element['entities'])  # entities를 dataframe으로 만든다.
            ele_id        = element['id']

            # 현재 relation에 대한 원인과 결과의 dict을 만들어주기 위해 반복문
            for _, val in relations_df.iterrows():
                self.process_ele_dict['type'].append(val['type'])  # relation으로 레이블링된 재난 종류 append
                copy_text_list = ele_text_list.copy()              # text를 list화한 것을 copy하여 다시 만들어줌 
               
                # 원인
                cause_dict  = entities_df[entities_df['id'] == val['from_id']].to_dict("records")[0] # Dataframe to dictionay 
                # 결과
                result_dict = entities_df[entities_df['id'] == val['to_id']].to_dict("records")[0]   # Dataframe to dictionay 
                
                entities_list  = [cause_dict, result_dict]                          # 해당 relation에 대한 원인 결과 dicationary 리스트
                copy_text_list = self.__insert_token(copy_text_list, entities_list) # 토큰이 추가된 리스트 만들기

                # 리스트로 만들어 둔 sentence를
                # 인과관계 토큰이 있는 sentence만 하나의 str로 만든다
                sentence_str = self.create_str_sentence(copy_text_list)
                self.process_ele_dict['text'].append(sentence_str) # "text" list에 추가

                # NIA 표준 json create
                if self.args.integrate : 
                    integrated_dict = self.insert_integrated_dict(val, ele_id, sentence_str)
                    self.INTEGRATED_DATA_DICT['data'].append(integrated_dict)

            pbar.update(1) # processbar update

        logger.info(f"======== Total Causal Count : {len(self.process_ele_dict['text'])}")

        return None


    """
    # 함수명  : insert_integrated_dict
    # 설명    : NIA 표준 json create
    # return  : self.init_integrate_dict
    # 특이사항 : 
    """
    def insert_integrated_dict(self, val, ele_id, sentence_str) :
        self.init_integrate_dict = self.INTEGRATED_DICT.copy()
        remove_token_sentence = re.sub(RegExpObject.E_OR_SLASH, "", sentence_str)
        self.init_integrate_dict[self.integrate_key_list[0]] = f"{uuid.uuid1().hex}_{ele_id}"        # id unique한 값을 가지기 위해 uuid1을 앞에 넣어줌
        self.init_integrate_dict[self.integrate_key_list[1]] = "화재조사보고서"                      # doc_loc # FIXME 값 바꿔줘야함
        self.init_integrate_dict[self.integrate_key_list[2]] = self.INTEGRATED_KEY_DICT[val['type']] # relation
        self.init_integrate_dict[self.integrate_key_list[3]] = remove_token_sentence                 # sentence

        def __insert_integrated_dict():
            e1_start = re.search(RegExpObject.E1,      sentence_str).span()[0]
            e1_end   = re.search(RegExpObject.E1_SLASH,sentence_str).span()[1]
            e2_start = re.search(RegExpObject.E2,      sentence_str).span()[0]
            e2_end   = re.search(RegExpObject.E2_SLASH,sentence_str).span()[1]
            
            # 토큰이 없는 sentence의 위치를 조정 하기 위해
            if e2_start < e1_start :
                e1_start = e1_start - self.redifine_position_val
                e1_end   = e1_end   - (self.redifine_position_val*2)
                e2_end   = e2_end   - self.redifine_position_val
            else : 
                e1_end   = e1_end   - self.redifine_position_val
                e2_start = e2_start - self.redifine_position_val
                e2_end   = e2_end   - (self.redifine_position_val*2)

            # dictionary에 추가
            self.init_integrate_dict[self.integrate_key_list[4]] = e1_start # subj_start
            self.init_integrate_dict[self.integrate_key_list[5]] = e1_end   # subj_end
            self.init_integrate_dict[self.integrate_key_list[6]] = e2_start # obj_start
            self.init_integrate_dict[self.integrate_key_list[7]] = e2_end   # obj_end
            self.init_integrate_dict[self.integrate_key_list[8]] = remove_token_sentence[e1_start:e1_end] # subj_word
            self.init_integrate_dict[self.integrate_key_list[9]] = remove_token_sentence[e2_start:e2_end] # obj_word

            return self.init_integrate_dict

        return __insert_integrated_dict()


    """
    # 함수명  : parse_args
    # 설명    : input parameter 
    # return  : args
    # 특이사항 : 
    """
    def parse_args(self):
        parser = argparse.ArgumentParser(
            description='Preprocess data input params')
                            
        parser.add_argument("-jlp", "--jsonl_path",    required=False, type=str, default=None, help="Data path")
        parser.add_argument("-tsp", "--tsv_save_path", required=False, type=str, default=None, help="result save path (only path!!)")

        true_false_list = ['true', 'yes', "1", 't','y']
        parser.add_argument("--tsv_save", 
                            type= lambda s : s.lower() in true_false_list, 
                            required=False, default=True, help="TSV save : True or False (e.g true,y, 1 | false, n, 0)")
        parser.add_argument("--integrate", 
                            type= lambda s : s.lower() in true_false_list, 
                            required=False, default=True, help="integrate json format : True or False (e.g true,y, 1 | false, n, 0)")

        # only create train, test, val use preprocessed tsv
        parser.add_argument("--tsv_path"  , required=False, type=str,  default=None, help="preprocessed tsv path (path & file name)")
        parser.add_argument("--train_json", required=False, type=bool, default=True, help="create for train But required tsv_path argument for use alone")

        args = parser.parse_args()

        return args


    """
    # 함수명  : main
    # 설명    : 데이터를 읽어서 전처리 후에 tsv를 저장한다.
    # return  : None
    # 특이사항 : 
    """
    def main(self):
        try : 
            # argument setting
            self.args = self.parse_args()

            # jsonl_path = "./data/oka1313@xaiplanet.com.jsonl" # 한개 짜리
            jsonl_path = "./data/oka1313@xaiplanet.com_원인n.jsonl" # 여러개 짜리

            # argument가 있을때 없을때를 구분!
            if self.args.jsonl_path :
                jsonl_path = self.args.jsonl_path
                logger.info(f"======== Input jsonl_path Argument : {jsonl_path}")
            
            if self.args.tsv_save_path :
                tsv_save_path = f"{self.args.tsv_save_path}/{os.path.splitext(os.path.basename(jsonl_path))[0]}_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.tsv"
                logger.info(f"======== Input save_path Argument : {tsv_save_path}")
            else : 
                tsv_save_path = f"{Paths.TSV_SAVE_DIR}/{os.path.splitext(os.path.basename(jsonl_path))[0]}_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.tsv"

            logger.info(f"======== Start Preprocessing ========")
            # create tsv and save
            data_lists   = self.read_jsonl(jsonl_path)           # read jsonl and 리스트화
            self.insert_causal_token_df(data_lists)              # 전처리 하여 tsv 만들기 위한 self.process_ele_dict 생성
            processed_df = pd.DataFrame(self.process_ele_dict)   # 전처리결과 DataFrame화
            # tsv 저장
            if self.args.tsv_save :
                createFolder(tsv_save_path)
                processed_df.to_csv(tsv_save_path, sep="\t", header=None, index=None)
                logger.info(f"======== Save TSV : {os.path.abspath(tsv_save_path)}")

            # create integrate json and save
            print(self.args.integrate,"===============================")
            if self.args.integrate :         # 통합 json create 유무 (argument : default = True)
                json_save_path = Paths.INTEG_JSON_DIR.format(datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
                createFolder(json_save_path)
                with open(json_save_path, 'w', encoding='utf-8') as f :
                    json.dump(self.INTEGRATED_DATA_DICT, f, ensure_ascii=False, indent=4)
                logger.info(f"======== Save Integrated json : {os.path.abspath(json_save_path)}")

            # 학습 위한 train test val 데이터 생성
            if self.args.tsv_path and self.args.train_json :
                CausalDataPreprocessor.main(self.args.tsv_path) # tsv save path
            elif self.args.train_json :
                CausalDataPreprocessor.main(tsv_save_path)      # tsv save path

            logger.info(f"======== Preprocessing  Done ========")

        except Exception : 
            err = traceback.format_exc(limit=4)
            logger.error(err)

        return None




if __name__ == "__main__" : 
    preprocessing = PreprocessJsonData()
    preprocessing.main()
    
