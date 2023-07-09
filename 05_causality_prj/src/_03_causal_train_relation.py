"""
# 파일명 : train_causal_relation.py
# 설명  : 인과관계 학습
# 개정이력 :
#    버젼    수정일자                   수정자              내용
#    1.0    2022-05-24                  김종완            신규 작성
""" 
import gc
import json
import math
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from random import randint
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from pandas import DataFrame
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning import Trainer as LightningTrainer
# from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils import column_or_1d
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import *
import re
import yaml
import mlflow

# common
from common import Params
from common import Paths
#logging setting
from common import Logging_config
logging_instance = Logging_config(Params.CUSTOM_LOG_FILE, Paths.LOG_DIR)
logger = logging_instance.logging_setting()

# processor
from processor import TrainParams
from processor import Train_config
from processor import GenericDataset



class BaseClassifier(LightningModule, ABC):
    """
    Base class of all classifiers
    """
    dataset_label_transform = None

    @abstractmethod
    def loss_function(self, logits: Tensor, label: Tensor) -> Tensor:
        """
        Calculate the loss of the model
        It MUST take care of the last activation layer
        """
        pass
    
    @abstractmethod
    def log_metrics(self, epoch_type: str, logits: Tensor, label: Tensor) -> dict:
        pass
    # TODO ④
    def __init__(self, pretrained_language_model, dataset_name, batch_size, learning_rate, decay_lr_speed,
                 dropout_p, activation_function, weight_decay, linear_size, keep_test_order):
        super().__init__()
        self.save_hyperparameters()
        self.test_proposed_answer = None
        self.keep_test_order      = keep_test_order

        self.language_model = AutoModel.from_pretrained(pretrained_language_model)
        config = self.language_model.config
        self.max_seq_len = config.max_position_embeddings
        self.hidden_size = config.hidden_size

        self.linear = nn.Linear(self.hidden_size, linear_size)
        self.linear_output = nn.Linear(linear_size, self.num_classes)

        self.dropout = nn.Dropout(p=dropout_p)
        self.activation_function = getattr(nn, activation_function)()
    
    # TODO ⑫
    def forward(self, sub_start_pos, sub_end_pos,
                obj_start_pos, obj_end_pos, *args, **kwargs) -> Tensor:
        language_model_output = self.language_model(*args, **kwargs)
        if isinstance(language_model_output, tuple):
            language_model_output = language_model_output[0]

        x = torch.mean(language_model_output, dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation_function(x)
        x = self.dropout(x)
        logits = self.linear_output(x)

        return logits

    def train_dataloader(self) -> DataLoader:
        return self.__get_dataloader("train")

    # TODO ⑦
    def val_dataloader(self) -> DataLoader:
        return self.__get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self.__get_dataloader("test")

    # TODO ⑧
    def __get_dataloader(self, subset: str) -> DataLoader:
        batch_size = self.hparams.batch_size
        dataset = GenericDataset(
            self.hparams.dataset_name,
            subset, 
            batch_size, 
            self.dataset_label_transform,
            self.keep_test_order
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1
        )
    # TODO ⑥
    def configure_optimizers(self):
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=float(self.hparams.learning_rate),
            weight_decay=self.hparams.weight_decay
        )
        # scheduler = LambdaLR(optimizer, lambda epoch: self.hparams.decay_lr_speed[epoch])
        scheduler = LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)
        return [optimizer], [scheduler]
    
    def training_step(self, batch: Tuple[dict, Tensor], batch_nb: int) -> dict:
        input_data, label = batch
        logits = self(**input_data)

        loss = self.loss_function(logits, label)
        log = {"train_loss": loss}

        return {"loss": loss, "log": log}

    # TODO ⑪-1
    def __eval_step(self, batch:  Tuple[dict, Tensor]) -> dict:
        input_data, label = batch
        # TODO ⑫-1
        logits = self(**input_data)

        return {
            "logits": logits,
            "label": label,
        }
    # TODO ⑪
    def validation_step(self, batch: Tuple[dict, Tensor], batch_nb: int) -> dict:
        return self.__eval_step(batch)
    
    def test_step(self, batch: Tuple[dict, Tensor], batch_nb: int) -> dict:
        return self.__eval_step(batch)

    def __eval_epoch_end(self, epoch_type: str, outputs: Iterable[dict]) -> dict:
        assert epoch_type in ["test", "val"]
        logits = torch.cat([x["logits"] for x in outputs]).cpu()
        label = torch.cat([x["label"] for x in outputs]).cpu()

        if epoch_type == "test" :
            print("============================",outputs)
        
        logs = self.log_metrics(epoch_type, logits, label)
        
        return {"progress_bar": logs}
    
    def validation_epoch_end(self, outputs: Iterable[dict]) -> dict:
        return self.__eval_epoch_end("val", outputs)

    def test_epoch_end(self, outputs: Iterable[dict]) -> dict:
        return self.__eval_epoch_end("test", outputs)
    
    def numeric_labels_to_text(self, label):
        """Revert labels from number to text"""
        if self.dataset_label_transform == "binary":
            label = ["Positive" if x else "Negative" for x in label]
        else:
            with open(Paths.METADATA_FILE_NAME, encoding='utf-8') as f:
                meta = json.load(f)[self.hparams.dataset_name]
            if self.dataset_label_transform == "none":
                mapping = meta["id_to_label"]
            else:
                mapping = meta["id_to_label_related_only"]
            label = [mapping[str(int(x))] for x in label]
        return label

    @staticmethod
    def plot_confusion_matrix(predicted_label, label) -> Figure:
        result = confusion_matrix(label, predicted_label)
        display = ConfusionMatrixDisplay(result)
        fig, ax = plt.subplots(figsize=(16, 12))
        display.plot(cmap=plt.cm.get_cmap("Blues"), ax=ax, xticks_rotation='vertical')
        return fig

    def log_confusion_matrix(self, prefix: str, predicted_label: Tensor, label: Tensor):
        predicted_label = self.numeric_labels_to_text(predicted_label)
        label = self.numeric_labels_to_text(label)
        fig = self.plot_confusion_matrix(predicted_label, label)
        
        # self.logger.experiment.log_figure(fig,f"./{prefix}_confusion_matrix.png")

        # self.logger.experiment.log_figure(fig,f"{prefix}_confusion_matrix")
        # self.logger.experiment.log_figure(fig,"./mlruns")
        # self.logger.experiment.log_image(f"{prefix}_confusion_matrix", fig)
        # self.logger.experiment.log_image(fig,f"{prefix}_confusion_matrix")
        # self.logger.experiment.log_image(fig,"./mlruns")




class MulticlassClassifier(BaseClassifier, ABC):
    """
    Base class for multiclass classifiers
    """

    def loss_function(self, logits: Tensor, label: Tensor)-> Tensor:
        return F.cross_entropy(logits, label)

    @staticmethod
    def logits_to_label(logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=-1)

    def log_metrics(self, epoch_type: str, logits: Tensor, label: Tensor) -> dict:
        predicted_label = self.logits_to_label(logits)

        self.log_confusion_matrix(epoch_type, predicted_label, label)

        if epoch_type =="test" :
            # print("\n label : ",label)
            # print("predicted_label : ",predicted_label)
            print("\n    input_label : ", self.numeric_labels_to_text(label))
            print("predicted_label : ", self.numeric_labels_to_text(predicted_label))

        # self.cnt = 0
        if epoch_type =="val" :
            # print("\n label : ",label)
            # print("predicted_label : ",predicted_label)
            # print("\n    input_label : ", self.numeric_labels_to_text(label))
            # print("predicted_label : ", self.numeric_labels_to_text(predicted_label))
            print("epoch : ",self.cnt)

            self.cnt +=1

            self.val_dict_df["input_label"].extend(self.numeric_labels_to_text(label))
            self.val_dict_df["predicted_label"].extend(self.numeric_labels_to_text(predicted_label))
            self.val_dict_df["epoch"].extend([self.cnt for i in range(len(self.numeric_labels_to_text(label)))])

            pd.DataFrame(self.val_dict_df).to_csv(f"./data/test_2_cnt_{self.cnt}.csv", encoding='utf-8', index=False)




        logs = {
            f"{epoch_type}_avg_loss"    : float(self.loss_function(logits, label)),
            f"{epoch_type}_acc"         : accuracy_score(label, predicted_label),
            f"{epoch_type}_pre_weighted": precision_score(label, predicted_label, average="weighted"),
            f"{epoch_type}_rec_weighted": recall_score(label, predicted_label, average="weighted"),
            f"{epoch_type}_f1_weighted" : f1_score(label, predicted_label, average="weighted"),
            f"{epoch_type}_pre_macro"   : precision_score(label, predicted_label, average="macro"),
            f"{epoch_type}_rec_macro"   : recall_score(label, predicted_label, average="macro"),
            f"{epoch_type}_f1_macro"    : f1_score(label, predicted_label, average="macro"),
        }

        # TODO end epoch에서 log 만들어 진거 뽑아 내는 듯?
        # print("==================logs",logs)

        # FIXME mlflow 세팅
        # with mlflow.start_run() as run:

        #     for k, v in logs.items():
        #         # self.logger.experiment.log_metric(k, v)
        #         print(k)
        #         print(v)
        #         self.logger.experiment.log_metric(key=k, value=v, run_id=run.info.run_id)
            # self.logger.experiment.log_metric(logs)


        # for k, v in logs.items():
        #     # self.logger.experiment.log_metric(k, v)
        #     print(k)
        #     print(v)
        #     self.logger.experiment.log_metric(key=k, value=v, run_id="0")
        # # self.logger.experiment.log_metric(logs)

        return logs




"""
# Class  : RelationClassifier
# 설명   : 
"""
class RelationClassifier(MulticlassClassifier):
    """
    A classifier that recognizes relations except for "not-related"
    """
    # TODO ③
    dataset_label_transform = "related_only"
    def __init__(self, dataset_name, **kwargs):
        with open(Paths.METADATA_FILE_NAME, encoding='utf-8') as f:
            self.num_classes = len(json.load(f)[dataset_name]["label_to_id_related_only"])
        
        self.cnt = 0
        self.val_dict_df = {
            "input_label" :[],
            "predicted_label" : [],
            "epoch":[],
        }
        super().__init__(dataset_name=dataset_name, **kwargs)


        # pd.DataFrame(self.val_dict_df).to_csv("./data/test_cnt.csv", encoding='utf-8', index=False)



"""
# Class  : Train_config
# 설명   : 
"""
# class CausalDataPreprocessor:
class CausalTrainprocessor:
    def __init__(self) :
        self.config_dict = Train_config.read_config() # config.yaml 파일을 읽어서 해당 내용 가져옴

        return
        

    def train(self):
        # rel_logger = MLFlowLogger(
        #     experiment_name="test_logger",
        # )

        with mlflow.start_run() as run:
            mlflow_uri = mlflow.get_tracking_uri()
            exp_id     = run.info.experiment_id
            exp_name   = mlflow.get_experiment(exp_id).name

            mlf_logger = MLFlowLogger(experiment_name=exp_name, tracking_uri=mlflow_uri)
            mlf_logger._run_id = run.info.run_id

            # trainer = pl.Trainer.from_argparse_args(args, logger=mlf_logger)

        try:
            
            rel_classifier = rel_trainer = None
            gc.collect()
            torch.cuda.empty_cache()
            # TODO ①
            rel_trainer = LightningTrainer(
                min_epochs = self.config_dict["REL_MIN_EPOCHS"],
                max_epochs = self.config_dict["REL_MAX_EPOCHS"],
                reload_dataloaders_every_epoch = True, # needed as we loop over a file,
                # reload_dataloaders_every_n_epochs=1, # MEMO torch > 1.5.1 일때
                default_root_dir    = Paths.CHECKPOINT_DIR,
                deterministic       = False,
                checkpoint_callback = True,
                # logger = rel_logger,                     # TODO logger 설정도 다시해야함
                logger = mlf_logger,                     # TODO logger 설정도 다시해야함
                gpus   = self.config_dict["GPUS"],
                # logger=mlflow.autolog()
            )
            # TODO ②
            rel_classifier = RelationClassifier(
                # pretrained_language_model = PRETRAINED_MODEL,
                pretrained_language_model = self.config_dict["pretrained_model_name"],
                activation_function       = self.config_dict["REL_ACTIVATION_FUNCTION"],
                dataset_name    = self.config_dict["DATASET_NAME"],  # FIXME 이거 없게 해야 할꺼 같은데..
                batch_size      = self.config_dict["REL_BATCH_SIZE"],
                learning_rate   = self.config_dict["REL_LEARNING_RATE"],
                decay_lr_speed  = self.config_dict["REL_LEARNING_RATE_DECAY_SPEED"],
                dropout_p       = self.config_dict["REL_DROPOUT_P"],
                weight_decay    = self.config_dict["REL_WEIGHT_DECAY"],
                linear_size     = self.config_dict["REL_LINEAR_SIZE"],
                keep_test_order = self.config_dict["keep_test_order"],
            )

            # TODO ⑤
            rel_trainer.fit(rel_classifier)
            rel_trainer.test(rel_classifier)

        except Exception as e:
        #     rel_logger.experiment.stop(str(e))
            raise e
        #     pass



if __name__ =="__main__" :
    causaltrain = CausalTrainprocessor()
    causaltrain.train()




