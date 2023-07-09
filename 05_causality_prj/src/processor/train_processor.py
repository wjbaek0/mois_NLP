"""
# 파일명 : train_processor.py
# 설명   : 학습을 위한 processor
# 개정이력 :
#    버젼    수정일자                   수정자              내용
#    1.0    2022-06-15                  김종완            신규 작성
""" 
import os
import yaml
import torch
import json
import math
from random import randint
from torch.utils.data import IterableDataset

#relation_processor
from processor import TrainParams

#common
from common import Paths



"""
# Class  : Train_config
# 설명   : 
"""
class Train_config :
    # global variable

    # MD change this variable to switch dataset in later tasks
    # TODO 이거 바꿔야 함
    DATASET_NAME = list(TrainParams.DATASET_MAPPING.keys())[0]

    
    """
    # 함수명   : read_config
    # 설명     : yaml파일을 읽어서 해당 모델에 대한 config dict return
    # return  : config_dict[model_name.upper()]
    # 특이사항 : 기본 kcbert의 config가 return 됨
    """  
    @staticmethod
    def read_config(model_name:str = "kcbert") :
        with open(Paths.CONFIG_FILE_DIR) as infile :
            config_dict = yaml.load(infile, Loader=yaml.FullLoader)
        
        return config_dict[model_name.upper()]


"""
# Class  : GenericDataset
# 설명   : 
"""
# TODO ⑨
class GenericDataset(IterableDataset):
    """A generic dataset for train/val/test data for both SemEval and GIDS dataset"""

    def __init__(self, dataset_name: str, subset: str, batch_size: int, label_transform: str, keep_test_order: bool):
        assert subset in ["train", "val", "test"]
        assert label_transform in ["none", "binary", "related_only"]

        # file_name = subset if label_transform == "none" \
        #     else f"{subset}_{label_transform}"

        file_name = Paths.TRAIN_VAL_TEST_PATH[subset]
        
        # preprocessor_class = get_preprocessor_class() # TODO 클래스 이름 가져오는거


        # with open(METADATA_FILE_NAME, encoding='utf-8') as f:
        #     metadata = json.load(f)[dataset_name]
        with open(Paths.METADATA_FILE_NAME, encoding='utf-8') as f:
            metadata = json.load(f)[dataset_name]

        # TODO ⑨-1 size 정해줌
        size = metadata[f"{subset}_related_only_size"] \
            if label_transform is "related_only" \
            else metadata[f"{subset}_size"]

        self.subset = subset
        self.batch_size = batch_size
        # TODO ⑨-2 length 계산
        self.length = math.ceil(size / batch_size)
        # TODO ⑨-3 file path 불러오기
        # self.file = open(preprocessor_class.get_dataset_file_name(file_name), encoding='utf-8')
        self.file = open(file_name, encoding='utf-8')

        # self.keep_test_order = self.subset == "test" and DATASET_MAPPING[dataset_name]["keep_test_order"]
        self.keep_test_order = self.subset == "test" and keep_test_order

    def __del__(self):
        if self.file:
            self.file.close()

    def __iter__(self):
        """
        Implement "smart batching"
        """

        data = [json.loads(line) for line in self.file]
        if not self.keep_test_order:
            data = sorted(data, key=lambda x: sum(x["attention_mask"]))

        new_data = []

        while len(data) > 0:
            if self.keep_test_order or len(data) < self.batch_size:
                idx = 0
            else:
                idx = randint(0, len(data) - self.batch_size)
            batch = data[idx:idx + self.batch_size]
            max_len = max([sum(b["attention_mask"]) for b in batch])

            for b in batch:
                input_data = {}
                for k, v in b.items():
                    if k != "label":
                        if isinstance(v, list):
                            input_data[k] = torch.tensor(v[:max_len])
                        else:
                            input_data[k] = torch.tensor(v)
                label = torch.tensor(b["label"])
                new_data.append((input_data, label))

            del data[idx:idx + self.batch_size]

        yield from new_data
    
    # def __len__(self):
    #     return self.length

    def as_batches(self):
        input_data = []
        label = []
        
        def create_batch():
            return (
                {k: torch.stack([x[k] for x in input_data]).cuda() for k in input_data[0].keys()},
                torch.tensor(label).cuda()
            )
        
        for ip, l in self:
            input_data.append(ip)
            label.append(l)
            if len(input_data) == self.batch_size:
                yield create_batch()
                input_data.clear()
                label.clear()

        yield create_batch()