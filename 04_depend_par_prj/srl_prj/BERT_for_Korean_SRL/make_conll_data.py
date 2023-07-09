'''
전체 보고서 요약된 데이터 summary_lexrank.json 를 불러와서  
문장만을 추출해 와서 text 파일을 형성하고,
train, valid, test .txt 형식을 .conll 데이터로 변환하기 위한 파일입니다.  
'''
from glob import glob
import glob
import json
from operator import index
from cv2 import line
from matplotlib.pyplot import text

import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import parsers
import re
import os
import ast

from jamo import h2j, j2hcj


#[함수1 - txt 파일로 변환하기 위한 함수] 
def make_total_text_data(input_json_file_name, output_text_file_name) :
    # 전체 보고서가 요약된 json 파일 입력 
    data = open(input_json_file_name, encoding = 'utf-8')
    data_dict = json.load(data)

    # json 파일에서 요약 데이터들이 들어있는 text 부분만 추출
    for i in data_dict :
        text = data_dict[i]['text']

        # text 파일로 저장
        f = open(output_text_file_name,'a', encoding='utf-8')
        for mystring in text :
            end_str = j2hcj(h2j(mystring))
            list_end_str = list(end_str)
            # 맨 마지막 문자가 '다' 와 'ㅁ'으로 끝나는 문장만 데이터로 사용
            if mystring[-1] == '다' or  list_end_str[-1] == 'ㅁ' :
                print(mystring)
                f.write(str(mystring))
                f.write('\n')

        f.close()
    
    return


#[함수2 - csv 파일로 변환하기 위한 함수] 
# output.txt 파일이 존재하는 경로와 출력하고자 하는 csv 파일 형태를 입력
def make_total_csv_data(text_file_path, output_csv) :

    p = parsers.srl_parser()
    r = re.compile('[^A-Za-z0-9가-힣\s]')
    
    with open(text_file_path, "r",encoding='cp949') as text :
        lines = text.readlines()
        temp_list = []

        for line in lines : 
            line = ' '.join(line.split())
            hangul = re.compile('[^ ㄱ-ㅣ가-힣 a-zA-Z,\d]+')
            line = hangul.sub("",line)
            
            sample_array = p.ko_srl_parser(line)
            temp_list.append(sample_array)
        

        global result_df  
        result_df = pd.DataFrame(temp_list).T
        result_df.to_csv(output_csv, encoding='utf-8', index=False)

    return result_df
    # 추후에 make_train_val_test_txt(result_df) 함수의 인자로 사용됨



#[함수 3 - train, valid, test set으로 데이터 분할을 위한 함수]
# 전체 csv 파일을 이용해 각각 train_data.txt : valid.txt : test.txt  = 8 : 1 : 1 로 나누고
# 각각의 csv로 저장 
def make_train_val_test_csv(output_csv) :

    # train : valid : test = 8 : 1 : 1 로 나누기 
    result_df = pd.read_csv(output_csv) 
    global train_df, valid_df, test_df
    train_df, test_df = train_test_split(result_df.T, test_size=0.2)
    valid_df, test_df = train_test_split(test_df, test_size=0.5)

    # 1972 sentence 
    print(train_df.shape) # 1577,43
    print(valid_df.shape) # 197,43
    print(test_df.shape) # 198,43

    train_df.to_csv("0728_train.csv", encoding='utf-8', index=False)
    valid_df.to_csv("0728_valid.csv", encoding='utf-8', index=False)
    test_df.to_csv("0728_test.csv", encoding='utf-8', index=False)

    return


#[함수4 - 텍스트를 리스트로 저장하여 train, valid, test = 8 : 1 : 1 로 나누어 주기]
def make_train_val_test_div(output_text_file_name):

    with open(output_text_file_name, 'r', encoding='UTF-8') as text :
        lines = text.readlines()
        train_list = []
        valid_list = []
        test_list = []
 
        # 문장들을 리스트에 담는다.
        for i,line in enumerate(lines) :
            if i < train_df.shape[0] : 
                train_list_file = open('0728_train.txt','a',encoding='utf-8')
                # 각각을 세개의 파일로 저장한다.
                train_list_file.write(line)

            elif train_df.shape[0] <= i and i < (train_df.shape[0] + valid_df.shape[0]) :
                valid_list_file = open('0728_valid.txt','a',encoding='utf-8')
                valid_list_file.write(line)

            else :
                test_list_file = open('0728_test.txt','a',encoding='utf-8')
                test_list_file.write(line)
            
    return 


#[함수 5 - 문장형태의 train.txt, valid.txt, test.txt 파일을 불러와 의미역분석 데이터에 맞게 변환을 위한 함수]
# make_total_csv_data(text_file_path, output_csv) 형태를 result_df 의 위치에 넣기
# 저장하고자 하는 파일명 .txt 포함하여 입력 , (txt 확장자로 저장 -> conll 로 변경) 
def make_train_val_test_txt(result_df, file_name):
 
    for i in range(result_df.shape[1]) : # result_df.shape[1] = 총 문장의 갯수  
        for j in range(result_df.shape[0]) : # result_df.shape[0] = 전체 문장 중 가장 서술어가 많은 문장의 갯수 
            try :
                # result_df 중 .conll data format 에 맞게 필요한 데이터만 출력
                f = open(file_name,'a',encoding='utf-8')
                result1 = result_df[i][j][0]
                result2 = result_df[i][j][1]
                result3 = result_df[i][j][2]  
                print('\n')

                try : 
                    for k in range(len(result_df[i][0][0])) :
                        result1[k] = result1[k].rstrip('\n') 
                        data = '\t'.join([str(k),result1[k],'PRED' if result2[k] != '_' else '_',result2[k],result3[k]])
                        f.write(data +'\n')
                        print('\t'.join([str(k),result1[k],'_',result2[k],result3[k]]))
                        
                except IndexError :
                    continue
                    
                f.write('\n')
                f.close()
            except TypeError :
                continue

    return


#[함수 6 - tran, valid, test 각각의 text 파일의 확장자를 conll 파일로 변환하기 위한 함수]
# conll 형태의 text 파일을 인자로 입력
def make_train_val_test_conll(text_file_name) :
    front_name = os.path.splitext(text_file_name)
    print(front_name[0]+'.conll')
    os.rename(text_file_name, front_name[0]+'.conll')
    
    return 


def main():
    make_total_text_data('data/summary_lexrank.json','data/output.txt')
    make_total_csv_data('D:\\WorkSpace\\SRL\\BERT_for_Korean_SRL_copy\\output_all_text.txt','output_all_text.csv')
    make_train_val_test_csv('output_all_text.csv')
    make_train_val_test_div('output_all_text.txt')
    make_train_val_test_txt(make_total_csv_data('D:\\WorkSpace\\SRL\\BERT_for_Korean_SRL_copy\\0818_train.txt','0818_train.csv'),'train_conll.txt')
    make_train_val_test_txt(make_total_csv_data('D:\\WorkSpace\\SRL\\BERT_for_Korean_SRL_copy\\0801_val.txt','0801_val.csv'),'val_conll.txt')
    make_train_val_test_txt(make_total_csv_data('D:\\WorkSpace\\SRL\\BERT_for_Korean_SRL_copy\\0801_test.txt','0801_test.csv'),'test_conll.txt')
    make_train_val_test_conll('train_conll.txt')
    make_train_val_test_conll('valid_conll.txt')
    make_train_val_test_conll('test_conll.txt')

if __name__=="__main__" :
    main()


'''

[함수1 - make_total_text_data]
전체 보고서가 요약된 summary_lexrank.json 자리에 요약된 파일 json 확장자로 입력
저장하고자 하는 파일명을 'output.txt' 자리에 txt 확장자로 입력
실행결과 : 'output.txt' 파일에 문장들이 추출되어 저장 
make_total_text_data('summary_lexrank.json','output.txt')


[함수2 - make_total_csv_data]
output.txt 파일이 존재하는 경로와 출력하고자 하는 csv 파일 형태를 입력
make_total_csv_data('D:\\WorkSpace\\SRL\\BERT_for_Korean_SRL_copy\\output.txt','output.csv')


[함수3 - make_train_val_test_csv]
만들고자 하는 output 파일 이름을 입력 
의미역 분석에 적합한 데이터 형태로 train_data : valid_data : test_data = 8 : 1 : 1 로 분배되어 csv 형태로 만들어짐
make_train_val_test_csv('output.csv')


[함수4 - read_list_fun]
output_text_file_name 입력
make_train_val_test_div('output.txt')


[함수5 - make_train_val_test_txt]
make_total_csv_data(실행할 text 파일경로, 데이터 프레임으로 저장할 csv파일 이름), conll 형식의 txt 파일로 저장할 이름 입력 
make_train_val_test_txt(make_total_csv_data('D:\\WorkSpace\\SRL\\BERT_for_Korean_SRL_copy\\0728_test.txt','0728_test.csv'),'0728_test_conll.txt')
valid, test 또한 동일하게 진행하기 


[함수6 - make_train_val_test_conll]
확장자 변경을 원하는 .txt 파일을 입력 
make_train_val_test_conll('0728_test_conll.txt')

'''