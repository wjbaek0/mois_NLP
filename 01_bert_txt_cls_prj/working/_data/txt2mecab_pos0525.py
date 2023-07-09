from konlpy.tag import Mecab
import glob
import os


def prep(pdf_file, save_path_mecab):
    # 토크나이저 선언
    tokenizer = Mecab(dicpath=r"C:\\mecab\\mecab-ko-dic")

    ## txt파일 불러오기 ##
    dt = open(pdf_file, 'r', encoding='utf-8')
    # txt파일 한 문장 씩 읽기
    ori_text = dt.readlines()

    # 파일 저장 위치 설정
    name = os.path.basename(pdf_file)
    wte = open(os.path.join(save_path_mecab, name), 'w', encoding='utf-8')

    # 형태소 분리(morph -> 품사태깅  pos 실행)
    for i in ori_text:
        morph = " ".join(tokenizer.morphs(i))  # 형태소 분리
        pos = tokenizer.pos(morph)    # 품사 태깅
        pos_pick = [i[0] for i in pos if i[1].find('NNP') != -1 or i[1].find(
            'NNG') != -1 or i[1].find('NNB') != -1 or i[1].find('VV') != -1]  # [,,,,,]

        # pos_pick : 품사에 NNP/NNG/NNB/VV가 들어가 있으면 리스트에 추가한다.
        # 출력 예시
        # ['화재', '공', '장동', '내', '압축', '공정', '시작']
        #['소', '재', '충청남도', '서산시', '소재']
        #['화재', '일시', '년', '월', '일', '수요일', '시', '분경']
        #['발화', '장소', '공', '장동', '압축', '공정']

        tokens = ' '.join(pos_pick)
        wte.write(tokens)
        wte.write('\n')  # 문장이 끝나면 개행문자 추가
    wte.close()


if __name__ == "__main__":

    # txt 파일 저장위치
    save_path_plumber = "C:/test_github/bert_classification/04.07/preprocessing/raw_files/plumber_raw/"

    # 형태소 분리파일 저장위치(결과 저장 위치)
    save_path_mecab = "C:/test_github/bert_classification/04.07/preprocessing/raw_files/mecab_raw/"
    os.makedirs(save_path_mecab, exist_ok=True)

    # txt 파일 경로
    save_path_lines = glob.glob(f"{save_path_plumber}*.txt")

    for num, i in enumerate(save_path_lines):
        prep(i, save_path_mecab)
        if num == 50 or num == 100 or num == 150 or num == 200:
            print(f'문장분리 중 {num}개 완료 ')
    print('## 문장분리 완료 ##')
