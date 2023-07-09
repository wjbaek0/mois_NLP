from sklearn.utils import column_or_1d
from collections import OrderedDict
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer
import pandas as pd
from pandas import DataFrame
from typing import Iterable, Tuple
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
class DataParing :
    TSV_NAMES = ["relation", "cause", "result", "sentence"]

    @staticmethod
    def search_token(series_data):
        regex = re.compile(r"\<\/*e\d\>")
        find_token_seq = regex.findall(series_data[DataParing.TSV_NAMES[3]]) # "sentence"

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
    DATASET_NAME = ""
    VAL_DATA_PROPORTION = 0.2
    NO_RELATION_LABEL = ""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        # TODO 스페셜 토큰 고려?? 
        # TODO 스페셜 토큰 추가 
        self.SUB_START_ID, self.SUB_END_ID, self.OBJ_START_ID, self.OBJ_END_ID \
            = tokenizer.convert_tokens_to_ids([TrainParams.SUB_OBJ_DICT['SUB_CHAR'][0], TrainParams.SUB_OBJ_DICT['SUB_CHAR'][1], # start, end
                                               TrainParams.SUB_OBJ_DICT['OBJ_CHAR'][0], TrainParams.SUB_OBJ_DICT['OBJ_CHAR'][1]] # start, end
                                               )
        self.label_encoder = OrdinalLabelEncoder([self.NO_RELATION_LABEL]) # TODO 이거 빼도 되지 않나 싶음

    # TODO 1.
    def preprocess_data(self, reprocess: bool):
        logger.info(f"\n---> Preprocessing {self.DATASET_NAME} dataset <---")

        # 폴더가 없다면 폴더 생성
        createFolder(self.PROCESSED_DATA_DIR)

        # stop preprocessing if file existed
        json_file_names = [self.get_dataset_file_name(k) for k in ("train", "val", "test")] # TODO 여기서 json 이름 만들어 주고
        existed_files = [fn for fn in json_file_names if os.path.exists(fn)]
        if existed_files:
            file_text = "- " + "\n- ".join(existed_files)
            if not reprocess:
                logger.info("The following files already exist:")
                logger.info(file_text)
                logger.info("Preprocessing is skipped. See option --reprocess.")
                return
            else:
                logger.info("The following files will be overwritten:")
                logger.info(file_text)

        train_data, val_data, test_data = self._preprocess_data()

        # TODO 그냥 기본 causaldata_???.json create
        # 이거 비효율적인듯
        logger.info("Saving to json files")
        self._write_data_to_file(train_data, "train")
        self._write_data_to_file(val_data, "val")
        self._write_data_to_file(test_data, "test")

        # TODO 메타데이터 만드는 부분
        self._save_metadata({
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "no_relation_label": self.NO_RELATION_LABEL,
            **self._get_label_mapping() # TODO _get_label_mapping
        })

        # TODO 중요!!!_create_secondary_data_files 파일 만드는 부분
        self._create_secondary_data_files()

        logger.info("---> Done ! <---")
    
    # TODO 3
    @abstractmethod
    def _preprocess_data(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        pass
    
    # TODO _create_secondary_data_files 메소드
    def _create_secondary_data_files(self):
        """
        From the primary data file, create a data file with binary labels
        and a data file with only sentences classified as "related"
        """

        with open(self.METADATA_FILE_NAME, encoding='utf-8') as f:
            root_metadata = json.load(f)
            metadata = root_metadata[self.DATASET_NAME]

        related_only_count = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        for key in ["train", "test", "val"]:
            logger.info(f"Creating secondary files for {key} data")
            
            # TODO 여기서 그냥 train 파일 만들걸 가지고 와서 다시 relation, binary를 만든다.
            origin_file = open(self.get_dataset_file_name(key), encoding='utf-8')
            bin_file = open(self.get_dataset_file_name(f"{key}_binary"), "w", encoding='utf-8')
            related_file = open(self.get_dataset_file_name(f"{key}_related_only"), "w", encoding='utf-8')

            total = metadata[f"{key}_size"]
            
            # TODO 이안에 key 갯수 만큼? 그니까 데이터 갯수만큼 돌린다??
            for line in tqdm(origin_file, total=total):
                data = json.loads(line)
                # TODO 그래서 위의 key키 tr, te, va 의 카운트를 친다??
                if data["label"] != 0:  # 0은 other
                    related_only_count[key] += 1
                    data["label"] -= 1 # label in "related_only" files is 1 less than the original label
                    related_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                    data["label"] = 1 # in binary dataset, all "related" classes have label 1
                    bin_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    bin_file.write(json.dumps(data, ensure_ascii=False) + "\n")

            origin_file.close()
            bin_file.close()
            related_file.close()

        logger.info("Updating metadata.json")
        for key in ["train", "test", "val"]:
            metadata[f"{key}_related_only_size"] = related_only_count[key]
        root_metadata[self.DATASET_NAME] = metadata
        
        # TODO 위에서 불러온거 다시씀 :155
        with open(self.METADATA_FILE_NAME, "w", encoding='utf-8') as f:
            json.dump(root_metadata, f, indent=4, ensure_ascii=False)

    # ===================================================================================================       
    def _create_secondary_data_files_modi(self):
        """
        From the primary data file, create a data file with binary labels
        and a data file with only sentences classified as "related"
        """
        with open(self.METADATA_FILE_NAME, encoding='utf-8') as f:
            root_metadata = json.load(f)
            metadata = root_metadata[self.DATASET_NAME]

        related_only_count = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        for key in ["train", "test", "val"]:
            logger.info(f"Creating secondary files for {key} data")
            
            # TODO 여기서 그냥 train 파일 만들걸 가지고 와서 다시 relation, binary를 만든다.
            origin_file = open(self.get_dataset_file_name(key), encoding='utf-8')
            bin_file = open(self.get_dataset_file_name(f"{key}_binary"), "w", encoding='utf-8')
            related_file = open(self.get_dataset_file_name(f"{key}_related_only"), "w", encoding='utf-8')

            total = metadata[f"{key}_size"]
            
            # TODO 이안에 key 갯수 만큼? 그니까 데이터 갯수만큼 돌린다??
            for line in tqdm(origin_file, total=total):
                data = json.loads(line)
                # TODO 그래서 위의 key키 tr, te, va 의 카운트를 친다??
                if data["label"] != 0:  # 0은 other
                    related_only_count[key] += 1
                    data["label"] -= 1 # label in "related_only" files is 1 less than the original label
                    related_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                    data["label"] = 1 # in binary dataset, all "related" classes have label 1
                    bin_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    bin_file.write(json.dumps(data, ensure_ascii=False) + "\n")

            origin_file.close()
            bin_file.close()
            related_file.close()

        logger.info("Updating metadata.json")
        for key in ["train", "test", "val"]:
            metadata[f"{key}_related_only_size"] = related_only_count[key]
        root_metadata[self.DATASET_NAME] = metadata
        
        # TODO 위에서 불러온거 다시씀 :155
        with open(self.METADATA_FILE_NAME, "w", encoding='utf-8') as f:
            json.dump(root_metadata, f, indent=4, ensure_ascii=False)
    # ===================================================================================================       



    def _find_sub_obj_pos(self, input_ids_list: Iterable) -> DataFrame:
        """
        Find subject and object position in a sentence
        """
        sub_start_pos = [self._index(s, self.SUB_START_ID) + 1 for s in input_ids_list]
        sub_end_pos = [self._index(s, self.SUB_END_ID, sub_start_pos[i]) for i, s in enumerate(input_ids_list)]
        obj_start_pos = [self._index(s, self.OBJ_START_ID) + 1 for s in input_ids_list]
        obj_end_pos = [self._index(s, self.OBJ_END_ID, obj_start_pos[i]) for i, s in enumerate(input_ids_list)]
        return DataFrame({
            "sub_start_pos": sub_start_pos,
            "sub_end_pos": sub_end_pos,
            "obj_start_pos": obj_start_pos,
            "obj_end_pos": obj_end_pos,
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

    def _write_data_to_file(self, dataframe: DataFrame, subset: str):
        """Write data in a dataframe to train/val/test file"""
        lines = ""
        for _, row in dataframe.iterrows():
            lines += row.to_json() + "\n"
        with open(self.get_dataset_file_name(subset), "w", encoding='utf-8') as file:
            file.write(lines)
    
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