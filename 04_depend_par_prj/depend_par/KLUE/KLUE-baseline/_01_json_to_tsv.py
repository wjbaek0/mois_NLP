import os, json, urllib3
import argparse
import pandas as pd
from konlpy.tag import Mecab

from koalanlp.Util import initialize, finalize
from koalanlp.proc import Tagger
from koalanlp import API



# 관형사 목록 출력 후 입력
# ['이', '한', '현', '다른', '두', '전', '여러', '약', '그', '모든', '세', '아무', '그런', '이런', '별', '제', '내', '동', '어떤', '총', '새', '어느', '저', '네', '첫', '각', 
#     '올', '모', '주', '옛', '타', '몇', '스무', '양', '맨', '연', '두세', '아무런', '단', '저런', '오랜', '무슨', '고', '만', '몇몇', '딴', '매', '양대', '온갖', '흔', '한두', '열네', '서너', '순']
MMA_LIST = ['온갖','새','헌','온','뭇','외딴','순','무려','단지','약','대략','별','각','단','오랜','맨','만','매']
MMD_LIST = ['이','그','저','요','고','조','이런','그런','저런','다른','어느','무슨','웬','옛','올','현','신','구','전','후','아무런','어떤','아무','모','타','이런저런','양대']
MM_LIST = []


MECAB_MM = False

def mecab_mm(sentence):
    sent_list = sentence.split()
    mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
    pos_tag = mecab.pos(sentence)
    for word in sent_list:
        for idx, tag in enumerate(pos_tag):
            if tag[1] == 'MM' and tag[0] not in MM_LIST:
                MM_LIST.append(tag[0])



# MECAB 형태소 분석기 사용
def mecab(sentence):
    sent_list = sentence.split()
    mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
    pos_tag = mecab.pos(sentence)
    print(pos_tag)
    tag_list = []
    n = 0
    for word in sent_list:
        new_word_list = []
        new_tag_list = []
        for idx, tag in enumerate(pos_tag):
            if idx < n:
                continue
            
            # 여기서 부터는 품사 태그 표 참고
            if tag[1] == 'SSO' or tag[1] == 'SSC':
                new_tag_list.append('SS')
            elif tag[1] == 'UNKNOWN':
                new_tag_list.append('NA')
            elif tag[1] == 'NNBC':
                new_tag_list.append('NNB')
            elif tag[1] == 'MM':
                tag_lem = tag[0]
                print(f'print MM {tag}')
                if tag_lem[-1] == '.':
                    tag_lem = tag_lem[:-1]

                if tag_lem in MMA_LIST:
                    new_tag_list.append('MMA')
                elif tag_lem in MMD_LIST:
                    new_tag_list.append('MMD')
                # MMA_LIST, MMD_LIST 외의 관형사들은 모두 MMN으로 분류하였음
                else:
                    new_tag_list.append('MMN')
            elif tag[1] == 'SC':
                new_tag_list.append('SP')
            elif tag[1] == 'SY':
                if tag[0] in ['O','X','~','□','-']:
                    new_tag_list.append('SO')
                else:
                    new_tag_list.append('SW')
            else:
                new_tag_list.append(tag[1])
            new_word_list.append(tag[0])

            # 어절이 일치하는 경우, 단어와 태그를 조합함
            if word == ''.join(new_word_list):
                n = idx+1
                tag_list.append((' '.join(new_word_list), '+'.join(new_tag_list)))
                new_word_list = []
                new_tag_list = []
                break
    return tag_list


'''
    국립국어원 데이터 -> klue 데이터 형태로 변환
'''
def mk_klue(args):

    # 국립국어원 데이터를 읽어옴
    with open(args.json_path, "r", encoding='utf-8-sig') as json_file :
        json_str = json.load(json_file)

    dataset_type = ['train', 'dev', 'test']
    for type in dataset_type:
        if type == 'train':
            num_start = 0
            num_data = int(args.num_of_sentence*0.8)
        elif type == 'dev':
            num_start = int(args.num_of_sentence*0.8)
            num_data = int(args.num_of_sentence*0.9)
        else:
            num_start = int(args.num_of_sentence*0.9)
            num_data = args.num_of_sentence

        error_path = os.path.splitext(args.json_path)[0] + f"_{type}_error.txt"
        ef = open(error_path, "w", encoding="utf-8")

        save_path=os.path.splitext(args.json_path)[0] + f"_{type}.tsv"
        with open(save_path, "w", encoding="utf-8") as wf:
            documents = json_str["document"]

            # 기존 데이터셋에서 형태를 참고
            dataset_path=f"./data/klue-dp-v1.1/klue-dp-v1.1_dev.tsv"
            with open(dataset_path, "r", encoding="utf-8") as rf:
                for idx, line in enumerate(rf):
                    # 상단 주석 다섯 줄 추가
                    if idx < 5: 
                        wf.write(line)

            sent_cnt = 0
            exit_bool = False
            
            # 국립국어원 데이터 형태 확인 필요
            for doc_idx, document in enumerate(documents):
                
                # 문서
                doc_id = document["id"]
                print(f"\n{doc_idx+1}번 문서 >> {doc_id}")
                
                sentences = document["sentence"]

                # 문장
                for sent_idx, sentence in enumerate(sentences):
                    sent_cnt+=1

                    if sent_cnt <= num_start:
                        continue

                    sentence_form = sentence["form"]
                    sentence_id = sentence["id"]
                    DP = sentence["DP"]
                    DP_df = pd.DataFrame(DP)

                    print(f"\n[ {len(sentences)} / {sent_idx+1} ] {sentence_form}")
                    
                    try:

                        # MECAB_MM = True : MECAB 관형사 목록을 출력
                        if MECAB_MM:
                            mecab_mm(sentence_form)

                            if num_start + (sent_cnt-num_start) == num_data:
                                exit_bool=True
                                break
                            continue

                        # MECAB_MM = False
                        else:
                            # mecab_tag_list = mecab_lemma(sentence_form)
                            mecab_tag_list = mecab(sentence_form)
                        # print(mecab_tag_list)

                    except:
                        exit()
                    
                    lemma_list = []
                    pos_list = []

                    # Mecab
                    for tags in mecab_tag_list:
                        pos_list.append(str(tags[1]))
                        lemma_list.append(str(tags[0]))
                        
                    # DP와 태깅 분석 결과가 일치하지 않을 경우, 오류 목록에 추가
                    if (len(lemma_list[-1]) == 1 and not lemma_list[-1].isalnum()) or len(DP_df) != len(lemma_list):
                        # lemma_list = lemma_list[:-2] + [' '.join(lemma_list[-2:])]
                        # pos_list = pos_list[:-2] + ['+'.join(pos_list[-2:])]
                        
                        print(f"DP와 태깅분석 결과 불일치 >> {lemma_list}")
                        ef.write(f"{doc_id}_{sentence_id} >> {lemma_list}\n")

                    try:
                        # DP 데이터프레임 열, 데이터 생성
                        DP_df.insert(2, 'lemma', lemma_list)
                        DP_df.insert(3, 'pos', pos_list)
                        print(f"누적 문장 수 >> {sent_cnt}")

                    except ValueError:
                        print(f"ValueError >> {lemma_list}")
                        ef.write(f"ValueError {doc_id}_{sentence_id} >> {lemma_list}\n")
                        continue

                    # 'dependent'는 KLUE-DP 형태에 포함되지않으므로 제거
                    DP_df.drop('dependent', axis=1, inplace = True)
                    # 컬럼명 재정의
                    DP_df.columns = ["INDEX", "WORD_FORM", "LEMMA", "POS", "HEAD", "DEPREL"]
                    # HEAD가 -1로 출력되는 부분(= ROOT)을 0으로 치환
                    DP_df.loc[DP_df["HEAD"] == -1, "HEAD"] = 0
                    
                    wf.write("## {}\n".format('\t'.join([f"{doc_id}_{sentence_id}", sentence_form])))

                    # 완성된 DP를 TSV 형태로 작성
                    for row in DP_df.values.tolist():
                        wf.write('\t'.join(map(str, row))+'\n')
                    wf.write('\n')    

                    # 문장 개수가 데이터셋별 개수와 크면 끝
                    if num_start + (sent_cnt-num_start) == num_data:
                        exit_bool=True
                        break

                if exit_bool:
                    break

        ef.close()

        # 관형사 목록이 존재할 경우 파일 작성
        if MM_LIST:
            print(MM_LIST)
            with open(os.path.splitext(args.json_path)[0] + f'_MM_{type}.txt', "w", encoding="utf-8") as wf:
                for mm in MM_LIST:
                    wf.write(mm+'\n')


def run(args):
    mk_klue(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert json to tsv")
    parser.add_argument("-jp","--json_path", required=False, type=str,   help="target json path", default="./data/국립국어원/NXDP1902103231.json")
    parser.add_argument("-nos","--num_of_sentence", required=False, type=int,   help="num of sentence", default="100")
    parser.add_argument("-mm","--mm", required=False, type=bool,   help="mecab for mm list", default=False)
    args = parser.parse_args()

    MECAB_MM = args.mm

    run(args)