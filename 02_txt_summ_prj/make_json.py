import os
import json
import argparse
import numpy as np
from lex_ranking import LexRanking  # 같은 위치의 lex_ranking.py를 불러옴

BASE_PATH = 'working'
DATA_PATH = BASE_PATH + '/_data'
SAVE_PATH = BASE_PATH + '/_result'


class MakeJson:
    def __init__(self):
        self.files = self.open_file()

    def open_file(self):
        with os.scandir(DATA_PATH) as entries:
            files = [entry.name for entry in entries if entry.is_file()]
        return files

class LexJson:
    def __init__(self):
        self.files = MakeJson().files
        self.lexrank_dict = self.make_lexrank_dict()

    def make_lexrank_dict(self):
        lexrank_dict = {}
        for file in self.files:
            lexrank = LexRanking(f'{DATA_PATH}/{file}')
            print(file)
            lexrank_dict[file] = lexrank.summary
        return lexrank_dict


if __name__ == "__main__":
    summary = LexJson().lexrank_dict
    iter = 1
    while True:
        if os.path.exists(f'{SAVE_PATH}/sum_lexrank_{iter}.json'):
            iter += 1
        else:
            with open(f'{SAVE_PATH}/sum_lexrank_{iter}.json', 'w', encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=4)
                f.close()
            break
    # print(f'Save to...\n{SAVE_PATH}')
