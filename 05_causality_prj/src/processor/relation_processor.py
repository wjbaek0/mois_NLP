"""
# 파일명 : relation_processor.py
# 설명   : 전처리를 위한 processor
# 개정이력 :
#    버젼    수정일자                   수정자              내용
#    1.0    2022-05-13                  김종완            신규 작성
""" 
from sklearn.utils import column_or_1d
from collections import OrderedDict
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer
import pandas as pd
from pandas import DataFrame
from typing import Iterable, Tuple, List
import os
from tqdm.auto import tqdm
import json
import re
#common
from common import Paths
from common import Params
from common import createFolder
#logging setting
from common import Logging_config
logging_instance = Logging_config(Params.CUSTOM_LOG_FILE, Paths.LOG_DIR)
logger = logging_instance.logging_setting()



"""
# Class  : DataParing
# 설명   : 
"""
class DataParsing :
    TSV_NAMES       = ["relation", "cause", "result", "sentence"]
    DF_TO_JSON_DICT = {
        "train" : "",
        "val"   : "",
        "test"  : "",
    }

    @staticmethod
    def search_token(series_data):
        regex = re.compile(r"\<\/*e\d\>")
        find_token_seq = regex.findall(series_data[DataParsing.TSV_NAMES[3]]) # "sentence"

        return find_token_seq


"""
# Class  : TrainParams
# 설명   : train시에 필요한 parameters 정의
"""
class TrainParams :
    # Data Mapping
    DATASET_MAPPING = {
        "CausalData": {
            "dir": Paths.CAUSAL_DATA_DIR,
            "keep_test_order": True,
            "precision_recall_curve_baseline_img": None,
        },
    }

    SUB_OBJ_DICT = {
        "SUB_CHAR" : ["[", "]"],
        "OBJ_CHAR" : ["{", "}"],
    }

    # BERT variants
    AVAILABLE_PRETRAINED_MODELS = [
        "bert-base-uncased",       
        "roberta-base",            
        "bert-large-uncased",      
        "bert-large-uncased",      
        "bert-large-uncased",      
        "beomi/kcbert-base",      
    ]


"""
# Class  : OrdinalLabelEncoder
# 설명   : # TODO OrdinalLabelEncoder 설명
"""
class OrdinalLabelEncoder:
    def __init__(self, init_labels=None):
        if init_labels is None:
            init_labels = []
        self.mapping = OrderedDict({l: i for i, l in enumerate(init_labels)})

    @property
    def classes_(self):
        return list(self.mapping.keys())

    def fit_transform(self, y):
        return self.fit(y).transform(y) # TODO 여기서 label 번호로 토큰 매핑해준 리스트 return

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        new_classes = pd.Series(y).unique()
        # TODO 위에 other로 mapping 만들어준거에 relation 추가
        for cls in new_classes:
            if cls not in self.mapping:
                self.mapping[cls] = len(self.mapping)
        return self

    def transform(self, y):
        y = column_or_1d(y, warn=True)
        return [self.mapping[value] for value in y]



"""
# Class  : AbstractPreprocessor
# 설명   : # TODO AbstractPreprocessor 설명
"""
class AbstractPreprocessor(ABC, Paths, TrainParams):
    DATASET_NAME        = ""   # TODO 이렇게 꼭 지정해서 해야 하나 싶다.
    VAL_DATA_PROPORTION = 0.2
    NO_RELATION_LABEL   = ""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        # TODO 스페셜 토큰 고려?? 
        # TODO 스페셜 토큰 추가 
        self.SUB_START_ID, self.SUB_END_ID, self.OBJ_START_ID, self.OBJ_END_ID \
            = tokenizer.convert_tokens_to_ids([TrainParams.SUB_OBJ_DICT['SUB_CHAR'][0], TrainParams.SUB_OBJ_DICT['SUB_CHAR'][1], # start, end
                                               TrainParams.SUB_OBJ_DICT['OBJ_CHAR'][0], TrainParams.SUB_OBJ_DICT['OBJ_CHAR'][1]] # start, end
                                               )
        self.label_encoder   = OrdinalLabelEncoder([self.NO_RELATION_LABEL]) # TODO 이거 빼도 되지 않나 싶음
        self.df_to_json_dict = DataParsing.DF_TO_JSON_DICT

    # TODO 1.
    def preprocess_data(self):
        logger.info(f"\n---> Preprocessing {self.DATASET_NAME} dataset <---")

        # 폴더가 없다면 폴더 생성
        createFolder(self.PROCESSED_DATA_DIR)
        train_data, val_data, test_data = self._preprocess_data() # go to _02_causal_data_integrate_preprocess_and_split.py  _preprocess_data

        # TODO 이름 바꿔야함.
        logger.info("Create to json format String")
        data_json_tp = self._write_data_to_file([train_data, val_data, test_data]) # return tuple format data

        # TODO 메타데이터 만드는 부분
        self._save_metadata({
            "train_size": len(train_data),
            "val_size"  : len(val_data),
            "test_size" : len(test_data),
            "no_relation_label": self.NO_RELATION_LABEL,
            **self._get_label_mapping() # TODO _get_label_mapping
        })

        self._create_secondary_data_files(data_json_tp) # create relation json 
        logger.info("---> Done ! <---")


    def _create_secondary_data_files(self, data_json_tp:Tuple[str,str,str]):
        """
        From the primary data file, create a data file with binary labels
        and a data file with only sentences classified as "related"
        """
        # read metadata.json
        with open(self.METADATA_FILE_NAME, encoding='utf-8') as f:
            root_metadata  = json.load(f)
            metadata       = root_metadata[self.DATASET_NAME]

        related_only_count = {
            "train": 0,
            "val": 0,
            "test": 0,
        }
        related_only_count_keys_list = list(related_only_count.keys())

        for key, str_json in zip(related_only_count_keys_list, data_json_tp) :
            logger.info(f"Creating relation files for {key} data")
            total = metadata[f"{key}_size"]
            str_json_list = str_json.split("\n")[:-1] # "\n"으로 split시에 마지막이 빈값임

            # relation data json save
            with open(self.TRAIN_VAL_TEST_PATH[key], "w", encoding='utf-8') as save_json :
                for line in tqdm(str_json_list, total=total) : 
                    data = json.loads(line)      # read str of dict format
                    related_only_count[key] += 1 # data count update
                    
                    # TODO 💡이부분 other가 빠지면 없어도 될듯?
                    data["label"] -= 1 # label in "related_only" files is 1 less than the original label

                    save_json.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            # relation count update in json            
            metadata[f"{key}_related_only_size"] = related_only_count[key]
        
        # change data in json
        logger.info("Updating metadata.json")
        root_metadata[self.DATASET_NAME] = metadata
        with open(self.METADATA_FILE_NAME, "w", encoding='utf-8') as f:
            json.dump(root_metadata, f, indent=4, ensure_ascii=False)


    def _write_data_to_file(self, df_data_list: List[DataFrame]) -> Tuple[str, str, str]:
        """순서 : train/val/test"""
        key_list = list(self.df_to_json_dict)
        for df, key in zip(df_data_list, key_list) :
            lines = ""
            for _, row in df.iterrows():
                lines += row.to_json() + "\n"

            self.df_to_json_dict[key] = lines

        return self.df_to_json_dict[key_list[0]], self.df_to_json_dict[key_list[1]], self.df_to_json_dict[key_list[2]]


    def _find_sub_obj_pos(self, input_ids_list: Iterable) -> DataFrame:
        """
        Find subject and object position in a sentence
        """
        sub_start_pos = [self._index(s, self.SUB_START_ID) + 1 for s in input_ids_list]
        sub_end_pos   = [self._index(s, self.SUB_END_ID, sub_start_pos[i]) for i, s in enumerate(input_ids_list)]
        obj_start_pos = [self._index(s, self.OBJ_START_ID) + 1 for s in input_ids_list]
        obj_end_pos   = [self._index(s, self.OBJ_END_ID, obj_start_pos[i]) for i, s in enumerate(input_ids_list)]
        return DataFrame({
            "sub_start_pos": sub_start_pos,
            "sub_end_pos"  : sub_end_pos,
            "obj_start_pos": obj_start_pos,
            "obj_end_pos"  : obj_end_pos,
        })

    @staticmethod
    def _index(lst: list, ele: int, start: int = 0) -> int:
        """
        Find an element in a list. Returns -1 if not found instead of raising an exception.
        """
        try:
            return lst.index(ele, start)
        except ValueError:
            return -1

    # TODO _clean_data 여기서 데이터 정합성으로 kcbert는 max가 300 이므로 초과는 뻄
    def _clean_data(self, raw_sentences: list, labels: list) -> DataFrame:
        if not raw_sentences:
            return DataFrame()
        # TODO 중요!!! 토큰화!!!
        tokens = self.tokenizer(raw_sentences, truncation=True, padding="max_length")

        data   = DataFrame(tokens.data)
        data["label"]    = self.label_encoder.fit_transform(labels)
        sub_obj_position = self._find_sub_obj_pos(data["input_ids"])
        data   = pd.concat([data, sub_obj_position], axis=1)
        data   = self._remove_invalid_sentences(data)
        return data

    def _remove_invalid_sentences(self, data: DataFrame) -> DataFrame:
        """
        Remove sentences without subject/object or whose subject/object
        is beyond the maximum length the model supports
        """
        # TODO max length 부분 지금 05_27 15:04 300으로 세팅되어 있음
        seq_max_len = self.tokenizer.model_max_length
        return data.loc[
              (data["sub_end_pos"] < seq_max_len)
            & (data["obj_end_pos"] < seq_max_len)
            & (data["sub_end_pos"] > -1)
            & (data["obj_end_pos"] > -1)
        ]
    # TODO _get_label_mapping 메소드
    def _get_label_mapping(self):
        """
        Returns a mapping from id to label and vise versa from the label encoder
        """
        # all labels
        id_to_label = dict(enumerate(self.label_encoder.classes_))
        label_to_id = {v: k for k, v in id_to_label.items()}

        # for the related_only dataset
        # ignore id 0, which represent no relation

        # TODO 딕셔너리 라벨과 id와 key value 교환식
        id_to_label_related_only = {k - 1: v for k, v in id_to_label.items() if k != 0}
        label_to_id_related_only = {v: k for k, v in id_to_label_related_only.items()}

        return {
            "id_to_label": id_to_label,
            "label_to_id": label_to_id,
            "id_to_label_related_only": id_to_label_related_only,
            "label_to_id_related_only": label_to_id_related_only,            
        }

    # TODO _save_metadata 메소드
    def _save_metadata(self, metadata: dict):
        """Save metadata to metadata.json"""
        # create metadata file
        if not os.path.exists(self.METADATA_FILE_NAME):
            logger.info(f"Create metadata file at {self.METADATA_FILE_NAME}")
            with open(self.METADATA_FILE_NAME, "w", encoding='utf-8') as f:
                f.write("{}\n")

        # add metadata
        logger.info("Saving metadata")
        with open(self.METADATA_FILE_NAME, encoding="utf-8") as f:
            root_metadata = json.load(f)
        with open(self.METADATA_FILE_NAME, "w", encoding='utf-8') as f:
            root_metadata[self.DATASET_NAME] = metadata
            json.dump(root_metadata, f, indent=4, ensure_ascii=False)
    
    @classmethod
    def get_dataset_file_name(cls, key: str) -> str:
        return os.path.join(Paths.PROCESSED_DATA_DIR, f"{cls.DATASET_NAME.lower()}_{key}.json")