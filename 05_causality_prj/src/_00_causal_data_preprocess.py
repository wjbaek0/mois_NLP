"""
# 파일명 : preprocess_data.py
# 설명   : 레이블링된 데이터 전처리 후 학습위한 tsv 파일 생성
# 개정이력 :
#    버젼    수정일자                   수정자              내용
#    1.0    2022-05-03                  김종완            신규 작성
""" 
import jsonlines
import os
import glob
import pandas as pd
import numpy as np
import re
import argparse
import traceback
from tqdm import tqdm  # processbar

# common
from common import DefineEntitiesName
from common import Params
from common import Paths
from common import createFolder

#logging setting
from common import Logging_config
logging_instance = Logging_config(Params.CUSTOM_LOG_FILE, Paths.LOG_DIR)
logger = logging_instance.logging_setting()



"""
# Class  : PreprocessJsonData
# 설명   : jsonl 데이터 token 추가 및 tsv형식으로 전처리
"""
class PreprocessJsonData(DefineEntitiesName):
    def __init__(self):
        self.process_ele_dict = self.INIT_DICT                      # common의 초기 빈값의 dictionary
        self.token_key_list   = list(self.CAUSAL_TOKEN_DICT.keys()) # ['원인', '결과']
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

        logger.info(f" ======== Labeling Text Data Count : {len(jsonl_element_list)}")

        return jsonl_element_list


    """
    # 함수명  : create_str_sentence
    # 설명    : 인과관계 토큰이 있는 문장만 만든다.
    # return  : sentence_str
    # 특이사항 : 정규식 사용
    """
    def create_str_sentence(self, ele_text_list):
        # TODO 문장은 이제 무조건 한문장이 되고 문장 나눔은 개행문자("\n")으로 바꾸는 작업
        # sentence_list = "".join(ele_text_list).split(".")
        sentence_list = "".join(ele_text_list).split("\n")

        regex = re.compile(r"\<\/*e\d\>") # 인과관계 토큰 정규식
        
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

        for element in jsonl_element_list : 
            ele_text_list = list(element['text'])              # 리스트화해서 한글자마다 인덱싱
            relations_df  = pd.DataFrame(element['relations']) # relations를 dataframe으로 만든다.
            entities_df   = pd.DataFrame(element['entities'])  # entities를 dataframe으로 만든다.

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

            pbar.update(1) # processbar update

        logger.info(f"======== Total Causal Count : {len(self.process_ele_dict['text'])}")

        return None


    """
    # 함수명  : parse_args
    # 설명    : input parameter 
    # return  : args
    # 특이사항 : 
    """
    def parse_args(self):
        parser = argparse.ArgumentParser(
            description='Preprocess data input params')
                            
        parser.add_argument("-p","--data_path", required=False, type=str, default=None, help="Data path")
        parser.add_argument("-s","--save_path", required=False, type=str, default=None, help="result save path")
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
            args = self.parse_args()

            # data_path = "./data/oka1313@xaiplanet.com.jsonl" # 한개 짜리
            data_path = "./data/oka1313@xaiplanet.com_원인n.jsonl" # 여러개 짜리
            save_path = "./data/test19_여러개.tsv"

            # argument가 있을때 없을때를 구분!
            if args.data_path :
                data_path = args.data_path
                logger.info(f"======== Input data_path Argument : {data_path}")
            if args.save_path :
                save_path = args.save_path
                logger.info(f"======== Input save_path Argument : {save_path}")

            logger.info(f"======== Start Preprocessing ========")
            data_lists = self.read_jsonl(data_path) # read jsonl and 리스트화
            self.insert_causal_token_df(data_lists) # 전처리 하여 tsv 만들기 위한 dictionary 생성
            
            # 전처리결과 DataFrame화
            processed_df = pd.DataFrame(self.process_ele_dict)
            # tsv 저장
            processed_df.to_csv(save_path, sep="\t", header=None, index=None)

            logger.info(f"======== End Preprocess & Save TSV : {os.path.abspath(save_path)} ========")

        except Exception : 
            err = traceback.format_exc(limit=4)
            logger.error(err)


        return None



if __name__ == "__main__" : 
    preprocessing = PreprocessJsonData()
    preprocessing.main()
    
