import argparse
import os
import csv
import pandas as pd
from glob import glob
import random
from transformers import BertTokenizer
from transformers import PreTrainedTokenizer



# 매개변수 지정 
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-p', '--path', type=str, default='D:\\새홀리기', help="dir")
parser.add_argument('-f', '--folder', type=str, default='mecab_pos', help="path for text file directory")
args = parser.parse_args()
dir = args.path
input = os.path.join(dir,args.folder)




# 토크나이저 설정 
tokenizer = BertTokenizer.from_pretrained(
    "beomi/kcbert-base",
    do_lower_case=False,
)

# 데이터 클래스 정의 
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ClassificationExample:
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

@dataclass
class ClassificationFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[int] = None

# 데이터 라벨 종류 지정 
def get_labels():
        return ["0", "1","2","3","4","5","6","7"]

@property
def num_labels(self):
    return len(self.get_labels())


#txt 파일에서 label, text 읽어 온 뒤 examples에 append 

def get_examples( data_root_path):
        lines = list(csv.reader(open(data_root_path, "r", encoding="utf-8"), delimiter="\t", quotechar='"'))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[1]
            label =  str(line[0])
            examples.append(ClassificationExample(text_a=text_a, text_b=None, label=label))

        return examples

# 토큰화 한 결과 출력  / 출력값 : 토큰화 된 문장 한줄로 출력
def convert_examples_to_classification_features(
        examples: List[ClassificationExample],
        tokenizer: PreTrainedTokenizer,
        label_list: List[str],
):
    label_map = {label: i for i, label in enumerate(label_list)}
    labels = [label_map[example.label] for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=128,
        padding="max_length",
        truncation=False,
    )
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = ClassificationFeatures(**inputs, label=labels[i])
        features.append(feature)
    sen = []
    for i, example in enumerate(examples):
            sen.append(tokenizer.convert_ids_to_tokens(features[i].input_ids))

    return sen



if __name__=='__main__':
    # 임의로 label을 달기 위해 txt 파일 출력 후 label random으로 부착 
    df = pd.DataFrame({'label':[],'text':[]})
    for i,dt in enumerate(glob(f"{input}\\*.txt")):
        my_file = open(dt, "r", encoding='utf-8')
        text_list = my_file.readlines()
        try:
            df.at[i, 'text'] = text_list[0]
        except:
            df.at[i, 'text'] = 'readline_error'
    label = [random.randint(0, 7) for i in range(len(df['text']))]
    df['label'] = label
    # label, sentence 형식의 csv 파일로 저장 
    df.to_csv(f"{dir}\\text_with_label.txt", sep = '\t', encoding = 'utf-8', index=False)


    # csv 파일 불러와 tokenize 
    csv.field_size_limit(100000000)
    examples = get_examples(os.path.join(dir,'text_with_label.txt'))
    result = convert_examples_to_classification_features(examples,tokenizer,label_list=get_labels())
    # 토크나이즈 한 문장의 길이정보를 담은 result_len 생성 
    result_len = [len(i) for i in result]
    # 데이터 프레임 생성 
    df_all = pd.DataFrame({'len':result_len,'text':result })
    # 데이터 프레임 csv 형식으로 저장 
    df_all.to_csv(f'{dir}\\text_result.txt',sep= '\t',encoding = 'utf-8', index = False)


    
