import os
from lexrankr import LexRank  # 패키지 수정하여 사용
from typing import List
from konlpy.tag import Mecab
import re

'''
이 소스에서는 하나의 파일에 대해서만 요약
여러 txt 파일을 요약하기 위해서는 여기를 거쳐 make_json.py에서 수행
'''

mecab = Mecab('C:\mecab\mecab-ko-dic')

BASE_PATH = 'data'
DATA_PATH = BASE_PATH + '/original'
SAVE_PATH = BASE_PATH + '/summary'


class LexRanking:
    def __init__(self, file):
        self.file = file
        self.text = self.open_file()
        self.lexrank = self.set_tokenizer()
        self.summary = self.generate_summary()
        self.sum_length, self.doc_length = self.summary['length'][0], self.summary['length'][1]

    def open_file(self):
        with open(self.file, encoding='utf-8') as f:
            text = f.read()
        return text

    def set_tokenizer(self):
        my_tokenizer: MecabTokenizer = MecabTokenizer()
        lexrank: LexRank = LexRank(my_tokenizer)
        return lexrank

    def generate_summary(self):
        self.lexrank.summarize(self.text)
        summary = self.lexrank.probe(0.2)
        summary['length'].append(self.lexrank.num_sentences)


        '''
        특수문자를 없애거나 치환하는 부분
        여기서는 요약이 끝난 문서에 대해 특수문자를 처리

        => 특정 단어가 들어있는 문장을 포함하거나 제외하는 부분은
        맨 위에서 언급한 lexrankr 패키지를 수정하여 요약 과정 안에서 이루어짐
        '''
        for i in range(len(summary['text'])):
            input_string = summary['text'][i]

            time_pattern = re.compile(r'([^\d]\d{2}) {0,3}[\:\：] {0,3}(\d{2})([^\d])')
            date_pattern = re.compile(r'([^\d]\d{2,4})\ {0,2}\.\ {0,2}(\d{1,2})\ {0,2}\.\ {0,2}(\d{1,2})([^\d])')
            comma_pattern = re.compile(r'(\d{1,})\,(\d{3})')
            space_pattern = re.compile(r'[\(\)\․\.\,\:\/\\]')
            pattern_punctuation = re.compile(r'[^\w\s\㎞\ｍ\㎜\㎡\㎠\㎖\ℓ\㏁\℃\%\×\~\～\-]')

            result = time_pattern.sub(r'\1시 \2분\3', input_string)
            result = date_pattern.sub(r'\1년 \2월 \3일\4', result)
            result = comma_pattern.sub(r'\1\2', result)
            result = space_pattern.sub(r' ', result)
            output_string = pattern_punctuation.sub(r'', result)

            summary['text'][i] = output_string

        return summary

    def print_summary(self):
        print('======== Summary ========', *self.summary['text'], sep='\n')
        print('\n======== Number of Sentences ========')
        print(
            f'Summarized: {self.sum_length} sentences / Original: {self.doc_length} sentences')


class MecabTokenizer:
    mecab: Mecab = Mecab('C:\mecab\mecab-ko-dic')

    def __call__(self, text: str) -> List[str]:
        # tokens: List[str] = self.mecab.pos(text, join=True)
        pos = self.mecab.pos(text, join=True)
        tokens: List[str] = [
            w for w in pos if '/NN' in w or '/VV' in w or '/VA' in w or '/XR' in w]
        return tokens


if __name__ == "__main__":
    file = input('확장자(.txt)를 제외한 파일명: ')
    file += '.txt'
    result = LexRanking(f'{DATA_PATH}/{file}')
    result.print_summary()
