from importlib.resources import read_text
import os
import re
import glob
from pathlib import Path
import hanja
import pandas as pd
import pdfplumber
from jamo import h2j, j2hcj
import pdftotext
from pykospacing import Spacing

"""
.pdf -> .txt 파일 변환  후 문장 분리  /  txt 변환 라이브러리 : pdfplumber
변환 순서 : pdf > 한자, 한글로 변환 후 pdfplumber_test 폴더 아래 .txt로 저장 
pdfplumber_test폴더의 .txt파일 읽어와서 문장 분리 후 같은폴더>같은 파일 명으로 저장 


예러 : from pykospacing import Spacing 실행에러 / GPU 메모리 부족 
"""


# pdf -> txt 변환
def text(pdfs_list, save_path):
    for path in pdfs_list:
        pdf = pdfplumber.open(path)
        pages = len(pdf.pages)
        name = os.path.splitext(os.path.basename(path))[0] + ".txt"
        #spacing = Spacing()
        f = open(save_path+'/'+name, 'w', encoding='UTF-8')

        for k in range(pages):  # 페이지의 모든 문자 개체를 단일 문자열로 조합, 개행문자 추가
            try:
                de_dupe_pages = pdf.pages[k].dedupe_chars(
                    tolerance=1)  # 동일 좌표상 중복글자 제거 ex)소소방방청청 1개 이상
                # 레이아웃 True 추출시 간략하게 정리해둔 카테고리가 겹치지 않음(간격 조절중)
                page = de_dupe_pages.extract_text(
                    x_tolerance=3, y_tolerance=3, layout=False, x_density=7.25, y_density=13)
                if page[:5] == '(cid:':
                    with open(path, "rb") as file:
                        page = pdftotext.PDF(file)[k]
                page = hanja.translate(page, 'substitution')
                page = spacing(page)
                f.write(page)
            except:
                # 중복서 오류 발생시 넘김
                page = pdf.pages[k].extract_text(
                    x_tolerance=3, y_tolerance=3, layout=False, x_density=7.25, y_density=13)
                if page[:5] == '(cid:':
                    with open(path, "rb") as file:
                        page = pdftotext.PDF(file)[k]
                page = hanja.translate(page, 'substitution')
                #page = spacing(page)
                f.write(page)
                continue

        f.close()


def prep(pdf_file, save_path_plumber):
    ## 파일 불러오기 ##
    dt = open(pdf_file, 'r', encoding='utf-8')  # txt파일 열기
    ori_text = dt.read().replace('\n', ' ')    # 줄바꿈 삭제
    text = ori_text

    ## 1. 글머리 번호 앞에 개행문자 추가하기 ##
    #   eg. 11. 3) (50) 등 글머리 번호가 나오면 앞에 \n 추가하기
    filter1 = re.compile(r' *\([0-9]+\) +')  # '(숫자) ' 형태 찾기 / eg. ' (3) '
    filter2 = re.compile(r' +[0-9]+\) +')    # ' 숫자) ' 형태 찾기 / eg. ' 30) '
    # ' 숫자. 문자'  형태 찾기 / eg. ' 45. 문장시작 ' -> 2022. 05. 25 와 같은 형태의 날짜의 문장구분을 막기 위해
    filter3 = re.compile(r' [0-9]+[.] +[^0-9]')
    index = [(m.start(0), m.end(0)) for filter in [filter1, filter2, filter3]
             for m in re.finditer(filter, ori_text)]  # 위 3가지 형태에 해당되는 str의 index 출력

    for i in range(len(index)):
        rep = ori_text[index[i][0]:index[i][1]]
        text = text.replace(rep, '\n'+rep)  # ' (3) ' 앞에 \n 추가
        # 띄어쓰기 중복 삭제 eg. '     3. 글머리' -> ' 3. 글머리'
        text = re.sub(' +', ' ', text)

    ## 2. 글머리 기호 앞에 개행문자 추가하기 ##
    # 아래의 글머리 기호가 나타나면 (가.~ 나.~ 라.~ ...) 개행문자 추가하기
    symbol = ['가.', '나.', '라.', '마.', '바.', '사.', '아.', '자.', '- ', ' <']

    ## 3. 특수문자 앞에 개행문자 추가하기 ##
    #  특수문자가 나타나면 앞에 개행문자 추가하기. 단, exception 리스트에 있는 특수문자는 개행문자를 추가하지 않는다.
    # exception에 해당하는 특수문자는 (㎢,㎖, &, <,(,:, ㎠) 등 문장구분에 해당하지 않는 특수문자이다.
    r = re.compile('[^A-Za-z0-9가-힣\s]')  # 특수문자만 찾기
    exception = ['㎉', '㎘', '㎢', ';', '〉', '〕', '⇒', '》', '℉', '÷', '％', '+', '㎖', '｜', '½', '㎧', '&', '<', '>', '(', ')', ':', '：', '」', '』', '‧', '․', 'ｍ', '㎡', '×', '㎏', '$', '“', '·', '㎜', '"', '”',
                 '①', '②', '‘', '’', ',', "'", '㎞', '㎸', '℃', '%', '~', '…', '?', '[', ']', '=', '/', '.', '∼', '-', '㎥', '@', '～', 'ℓ', '㎝', '→', '', '】', '/', '㎠', '㎛', 'Ⅲ', 'Ⅱ']   # 리스트 안에 있는 특수문자는 줄바꿈 하지 않습니다.

    # text파일의 모든 글자를 순서대로 검색하기
    txt = []
    name = os.path.basename(pdf_file)
    for num, k in enumerate(text):   # 한글자씩 검색하기

        # <2.번 실행>
        if text[num:num+2] in symbol:  # < 가. 나. 라. 마. 바. 사. 아. 자. '- ',' <' >가 나오면
            k = '\n' + k                                              # 줄바꿈

        # <3.번 실행>
        if r.search(k) is not None:  # 글자가 특수문자고
            word = r.search(k).group()
            if word not in exception:   # 예외 리스트에 들어가 있지 않으면
                # 줄바꿈 + 문자로 변경 -> 특수문자 앞에 개행문자 부착
                k = k.replace(word, '\n'+word)

    ## 4. 다.로 끝나면 개행문자 추가  ##
        if text[num-1:num+2] == '다. ':
            k = k+'\n'

    # 5. -ㅁ. (명사형 어미)/ ~것. 으로 끝나면 개행문자 추가
        if num != 0:
            # 임. 음. 함. / 것. 으로 끝나면
            last = (j2hcj(h2j(text[num-1]))[-1], j2hcj(h2j(text[num]))[-1])
            if last == ('ㅁ', '.') or last == ('ㅅ', '.'):                  # 줄바꿈
                k = k + '\n'

        # 변경한 요소 리스트에 저장
        txt.append(k)
        txt_lines = "".join(txt)   # 리스트를 str로 변환

    # 같은 위치, 같은 이름의 파일로 저장
    with open(os.path.join(save_path_plumber, name), 'w', encoding='utf-8')as f:
        f.write(txt_lines)


if __name__ == "__main__":
    # pdfs_list = ["./Users/dami/Desktop/pdfplumber_test/01. 충주 화학공장 화재.pdf"]
    # pdf 파일 위치
    pdfs_list = glob.glob(
        "C:/test_github/bert_classification/04.07/preprocessing/test/new_pdf/*.pdf")

    # txt파일 저장위치
    save_path_pdf_to_txt = "C:/test_github/bert_classification/04.07/preprocessing/test/txt_raw/"

    # plumber 파일 저장위치
    save_path_plumber = "C:/test_github/bert_classification/04.07/preprocessing/test/plumber_raw/"

    os.makedirs(save_path_pdf_to_txt, exist_ok=True)
    os.makedirs(save_path_plumber, exist_ok=True)

    text(pdfs_list, save_path_pdf_to_txt)
    print('## plumber사용 txt변환 완료 ##')

    # txt파일 저장 폴더 내 모든 파일
    save_path_lines = glob.glob(f"{save_path_pdf_to_txt}*.txt")

    for num, i in enumerate(save_path_lines):
        prep(i, save_path_plumber)
        if num == 50 or num == 100 or num == 150 or num == 200:
            print(f'문장분리 중 {num}개 완료 ')
    print('## 문장분리 완료 ##')
