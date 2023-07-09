"""
# 파일명 : _01_causal_data_preprocess.py
# 설명  : tsv에서 학습을 위한 json파일로의 변환
# 개정이력 :
#    버젼    수정일자                   수정자              내용
#    1.0    2022-05-24                  김종완            신규 작성
"""
from typing import Tuple
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
import re
import os
import argparse
import traceback
# common
from common import Params
from common import Paths
#logging setting
from common import Logging_config
logging_instance = Logging_config(Params.CUSTOM_LOG_FILE, Paths.LOG_DIR)
logger = logging_instance.logging_setting()

# processor
from processor import TrainParams
from processor import AbstractPreprocessor
from processor import DataParsing



class CausalDataPreprocessor(AbstractPreprocessor):
    DATASET_NAME      = "CausalData"
    NO_RELATION_LABEL = "Other"

    def __init__(self, tokenizer: PreTrainedTokenizer, train_data_path, test_data_path):
        super().__init__(tokenizer)
        self.read_train = self.read_tsv(train_data_path)
        self.read_test  = self.read_tsv(test_data_path) # TODO 우선 밑에서 쓰진 않음
    

    def read_tsv(self, data_path):
        raw_tsv_df = pd.read_csv(data_path, sep="\t",encoding="utf-8", header=None, names=DataParsing.TSV_NAMES) # DataParsing.TSV_NAMES : list
        return raw_tsv_df
                                      

    # TODO 3.
    def _preprocess_data(self):
        # train test split
        split_train_df, split_test_df = train_test_split(self.read_train, shuffle=True, random_state=42, test_size=0.2)
        logger.info("Processing training data")
        train_data = self._process_file(split_train_df)

        logger.info("Processing test data")
        test_data = self._process_file(split_test_df)

        logger.info("Splitting train & validate data")
        train_data, val_data = train_test_split(train_data, shuffle=True, random_state=42, test_size=0.1)

        return train_data, val_data, test_data
    

    def _process_file(self, file_name) -> DataFrame:
        raw_sentences = []
        labels = []

        for _, val in file_name.iterrows() : 
            sent = val[DataParsing.TSV_NAMES[3]] # "sentence"
            label, sub, obj = self._process_label(val)
            labels.append(label)
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
        nums = list(filter(str.isdigit, str(find_token_seq))) # 원인과 결과 번호 구분 

        return label, int(nums[0]), int(nums[2])

    
    """
    # 함수명  : parse_args
    # 설명    : input parameter 
    # return  : args
    # 특이사항 : 
    """
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(
            description='causal data input params')
                            
        parser.add_argument("-trp","--train_path", required=False, type=str, default=None, help="train data path")
        parser.add_argument("-tep","--test_path",  required=False, type=str, default=None, help="test data path")
        parser.add_argument("-s","--save_path",    required=False, type=str, default=None, help="result save path")
        args = parser.parse_args()

        return args
   

    @classmethod
    def main(cls):
        train_data_path = "./data/test19_여러개.tsv"
        test_data_path  = "./data/test19_여러개.tsv"
        arg = cls.parse_args() # parse args define
        
        if arg.train_path :
            train_data_path = os.path.abspath(arg.train_path)
            logger.info(f"input argument train data path : {train_data_path}")
        if arg.test_path :
            test_data_path = os.path.abspath(arg.test_path)
            logger.info(f"input argument test data path  :{test_data_path}")


        pretrained_model = TrainParams.AVAILABLE_PRETRAINED_MODELS[5] # kcbert

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        # some tokenizer, like GPTTokenizer, doesn't have pad_token
        # in this case, we use eos token as pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # processor 실행
        preprocessor = CausalDataPreprocessor(tokenizer       = tokenizer,
                                              train_data_path = train_data_path,
                                              test_data_path  = test_data_path
                                             )
        preprocessor.preprocess_data(reprocess=True)

        return None



if __name__ == "__main__" : 
    CausalDataPreprocessor.main()