# 인과관계 소스코드 

## 디렉토리 구조

```bash
├── common
│   ├── __init__.py
│   ├── common.py
│   └── log_config.py
├── processor
│   ├── __init__.py
│   ├── relation_processor.py
│   └── train_processor.py
├── _00_causal_data_preprocess.py
├── _01_causal_data_preprocess.py
└── _02_causal_data_integrate_preprocess_and_split.py
```

## 디렉토리 구조 설명

|폴더    |폴더/파일명                                      |파일                 |설명                                                          |
|:------:|:------------------------------------------------|:--------------------|-------------------------------------------------:            |
|src     |                                                 |                     |                                                              |
|        |common                                           |                     |                                                              |
|        |                                                 | \_\_init\_\_.py     | common.py에서 정의한 class 및 함수 정의                      |
|        |                                                 | common.py           | 공통으로 쓸수 있는 class 및 함수 정의                        |
|        |                                                 | log_config.py       | log에 대한 config 정의                                       |
|        |processor                                        | \_\_init\_\_.py     | processor 프로그램에서 정의한 class 및 함수 정의             |
|        |                                                 |relation_processor.py| 전처리시에 필요한 class 및 함수 정의                         |
|        |                                                 |train_processor.py   | 학습을 위한 class 및 함수 정의                               |
|        |_00_causal_data_preprocess.py                    |                     | doccano 추출물 jsonl 데이터 token 추가 및 tsv형식으로 전처리 |
|        |_00_causal_data_preprocess.py                    |                     | 학습을 json생성(tsv --> json)                                |
|        |_02_causal_data_integrate_preprocess_and_split.py|                     | 모든 천처리 과정 통합                                        |
