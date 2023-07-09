"""
# 파일명 : log_config
# 설명   : logging define
# 개정이력 :
#    버젼    수정일자                   수정자              내용
#    1.1    2022-05-09                  김종완            신규작성
""" 
import logging
import logging.config
import json
import os



"""
# Class  : Logging_config
# 설명    : log setting
"""
class Logging_config:
    def __init__(self, create_dir=False, file_name="causal_log.log"):
        self.file_name  = file_name
        self.create_dir = create_dir


    """
    # function : config_define
    # 설명    : log config를 정의하고 file name을 변경
    # params : file_name
    # return : config_dict
    """
    def config_define(self):
        config_dict={
            "version" : 1,
            "disable_existing_loggers" : False,
            "formatters" : {
                "default" : {
                    "format" : "[%(asctime)s](%(filename)s:%(lineno)d)[%(levelname)s] %(message)s",
                    "datefmt" : "%Y-%m-%d %H:%M:%S"
                },
                "error" : {
                    "format" : "[%(asctime)s](%(filename)s:%(lineno)d)[%(levelname)s] %(message)s",
                    "datefmt" : "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers" : {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "DEBUG",
                    "formatter": "default"
                },
            },
            "loggers": {
                "logger": {
                    "level": "DEBUG",
                    "handlers": ["console"], # console
                    "propagate": True
                },
            }
        }

        # 파일 및 dir 생성 True시에
        if self.create_dir : 
            RotatingFileHandler = {
                        "RotatingFileHandler" :{
                            "class": "logging.handlers.RotatingFileHandler",
                            "filename" : self.file_name,
                            "mode" :"a",
                            "maxBytes" : 1024*1024*10,
                            "backupCount" : 1,
                            "encoding" : "utf-8",
                            "delay" : False,
                            "formatter": "default"
                            }
                        }
            config_dict['handlers'].update(RotatingFileHandler)
            config_dict['loggers']['logger']['handlers'].append("RotatingFileHandler")

        return config_dict
   
   
    """
    # function : logging_setting
    # 설명    : log폴더가 없다면 생성하고, logger create
    # params : file_name
    # return : logger
    """
    def logging_setting(self):
        # 폴더가 없다면 생성
        if self.create_dir : 
            if not os.path.exists(os.path.dirname(self.file_name)):
                os.makedirs(os.path.dirname(self.file_name))

        # config define and file name define
        config_dict = Logging_config.config_define(self)

        # logging config 적용
        logging.config.dictConfig(config_dict)

        # create logger
        logger = logging.getLogger("logger")

        return logger