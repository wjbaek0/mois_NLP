# KLUE Baseline

[KLUE](https://klue-benchmark.com/) Basline 모델 학습을 위한 저장소입니다. KLUE에 포함된 각 과제의 Baseline 모델의 자세한 내용은 [논문](https://arxiv.org/pdf/2105.09680.pdf)을 참조하시기 바랍니다.

#### 하위 디렉토리 설명

| 폴더명                 | 설   명                                |
| ---------------------- | -------------------------------------- |
| data                   | 데이터                                 |
| klue/roberta-base      | KLUE-DP 데이터로 프리트레인 된 모델    |
| klue_baseline          | KLUE Baseline 코드                     |
| output                 | 수행결과 저장                          |
| _01_json_to_tsv.py     | json 형식의 데이터를 tsv 형식으로 변환 |
| _02_run_klue.py        | 학습 및 검증                           |
| _03_infer_with_pred.py | 추론 결과 출력                         |
| run_all.sh             | 전체 task 실행 (dp 이외의 task도 있음) |
| requirements.txt       | 실행환경                               |




## Dependencies

KLUE Basline 모델 학습에 필요한 라이브러리는 requirements.txt 에 있습니다. 설치하려면 아래와 같이 실행하시면 됩니다.

```bash
pip install -r requirements.txt
```

모든 실험은 Python 3.7 기준으로 테스트되었습니다.



## Data

>- 학습에 사용한 데이터는 아래 경로에서 확인 가능
>  - \\192.168.219.150\XaiData\AI사업본부\99.인수인계\전유리\행안부_의존구문분석\KLUE_DATA : 프리트레인에 사용된 KLUE에서 제공하는 데이터_
>  - _\\192.168.219.150\XaiData\AI사업본부\99.인수인계\전유리\행안부_의존구문분석\국립국어원_모두의말뭉치_DATA : 학습에 사용할 데이터
>
>- KLUE-DP 데이터셋 참고를 위해 ‘data’ 디렉토리에 저장한다. -> train, val, test 셋 중 아무거나 상관 없음. 형태가 같음.
>
>    국립 국어원 데이터를 ‘data’ 디렉토리에 저장한다.
>
>  ![image-20220907213408641](C:\Users\yuri\AppData\Roaming\Typora\typora-user-images\image-20220907213408641.png)



## Model

> - 프리트레인된 모델은 아래 경로에서 확인 가능
>   - \\192.168.219.150\XaiData\AI사업본부\99.인수인계\전유리\행안부_의존구문분석\MODEL
>   - "config.json": 언어모델의 구조 정보를 담고 있는 파일 
>   - "pytorch_model.bin": 언어모델 weight를 저장한 파일 
>   - "tokenizer_config.json": 토크나이저 구성 정보 파일 




## Dataset

```bash
python _01_json_to_tsv.py -jp {} -nos {} -mm {}
```

> 1. 실행방법 : 아래의 파라미터 수정 후 위의 명령어 실행
>
>    - json_path(jp) : 대상 json 데이터의 경로를 입력(default : ./data/국립국어원/NXDP1902103231.json’)
>
>    - num_of_sentence(nos) : 총 문장 개수 입력 (default : 100)
>
>    - mm : 관형사 목록 출력 모드 사용 여부(default : False)



## Train

```bash
python _02_run_klue.py train --task klue-dp --output_dir {} --data_dir {DATA_DIR}/{task}-{VERSION}  --model_name_or_path {} --learning_rate {} --num_train_epochs {} --warmup_ratio {} --train_batch_size {} --patience {} --max_seq_length {} --metric_key las_macro_f1 --gpus {} --num_workers {}
```

> 1. 실행방법 : 아래의 파라미터 수정 후 위의 명령어 실행
>
>    - task : 태스크 종류( ‘klue-dp’ 고정)
>
>    - output_dir : 결과 저장 디렉토리 경로
>
>    - data_dir : 학습 데이터 디렉토리 경로
>
>    - model_name_or_path : 모델 경로 (default : ‘../../klue/roberta-base’)
>
>    - learning_rate : 러닝 레이트 (default : 5e-5)
>
>    - num_train_epochs : 최대 에포크 수 (default : 4)
>
>    - warmup_ratio : 웜업 스텝 비율. 현재 step이 warmup_step보다 낮을 경우는 learning rate를 linear하게 증가 시킴. warmup_step = total_step / warmup_ratio
>
>    - train_batch_size : 배치사이즈 (default : 32)
>
>    - patience : 향상되지 않으면 해당 validation 에포크 수 이후에 학습이 멈춤 (default : 5)
>
>    - max_seq_length : 토큰화 후 최대 총 입력 시퀀스 길이. 이보다 긴 시퀀스는 잘리고 짧은 시퀀스는 패딩됨(default : 128)
>
>    - metric_key : 모니터링할 매트릭스의 이름
>
>    - gpus : gpu 사용 여부
>
>    - num_workers : 학습 도중 CPU의 작업을 몇 개의 코어를 사용해서 진행할지 설정 (default : 4)
>
>    - 이외 설정 가능한 파라미터들 다수 존재
>
>      
>
> 2. 참고
>
>    - ![image-20220907214029626](C:\Users\yuri\AppData\Roaming\Typora\typora-user-images\image-20220907214029626.png)
>
>    - 학습을 진행하면서 1epoch당 4번의 evaluate 진행
>    - 학습이 완료되면 자동으로 best로 evaluate 진행



## Evaluate

```bash
python _02_run_klue.py test --task klue-dp --output_dir {OUTPUT_DIR} --data_dir {DATA_DIR}/{task}-{VERSION}  --model_name_or_path {model_name} --gpus {}
```

> 1. 실행방법 : 아래의 파라미터 수정 후 위의 명령어 실행
>    - task : 태스크 종류( ‘klue-dp’ 고정)
>    - eval_batch_size : 배치사이즈 (default : 64)
>    - 나머지는 train과 동일
> 2. 실행결과
>    - output\klue-dp\version_{버전} 경로 하위에 저장됨
>    - version_{버전} 하위 ![image-20220907212709493](C:\Users\yuri\AppData\Roaming\Typora\typora-user-images\image-20220907212709493.png)
>    - transformers 하위 ![image-20220907212723595](C:\Users\yuri\AppData\Roaming\Typora\typora-user-images\image-20220907212723595.png)
>    - pred 하위 <img src="C:\Users\yuri\AppData\Roaming\Typora\typora-user-images\image-20220907212931312.png" alt="image-20220907212931312" style="zoom: 80%;" />



## Inference

```bash
python _03_infer_with_pred.py -v {} -dt {}
```

> 1. 실행방법 : 아래의 파라미터 수정 후 위의 명령어 실행
>
>    - version(v) : 대상 json 데이터의 경로를 입력(default : 30)
>    - dataset_type(dt) : 데이터셋 종류 입력(default : ‘train’)
>
> 2. 실행결과
>
>    - output\klue-dp\version_{버전}\transformers\pred\pred-0.json
>
>    - <img src="C:\Users\yuri\AppData\Roaming\Typora\typora-user-images\image-20220907212139468.png" style="zoom:80%;" /><img src="C:\Users\yuri\AppData\Roaming\Typora\typora-user-images\image-20220907213030321.png" alt="image-20220907213030321" style="zoom:80%;" />
>
>    - head_preds : 헤드(예측)
>
>      type_preds : 의존관계태그(예측)
>
>      head_labels : 헤드(정답)
>
>      type_labels : 의존관계태그(정답)
>
>    - ![image-20220907213129811](C:\Users\yuri\AppData\Roaming\Typora\typora-user-images\image-20220907213129811.png)





## Reference

이 저장소의 코드나 KLUE 데이터를 활용하신다면, 아래를 인용 부탁드립니다.
```
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation}, 
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
