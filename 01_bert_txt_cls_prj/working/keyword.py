from konlpy.tag import Kkma, Komoran, Mecab
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt


def keyword(data, dicpath):
    mecab = Mecab(dicpath=dicpath)
    with open(data, 'r', encoding='utf-8') as f:
        file = f.read()
        nouns = mecab.nouns(file)
        count = Counter(nouns)
        most_common = count.most_common()
        keyword = [word[0] for idx in range(len(most_common)) for word in most_common]
        wc = WordCloud(font_path=r'C:/Windows/Fonts/malgun.ttf', background_color="white", max_font_size=60)
        cloud = wc.generate_from_frequencies(count)
        # wc.to_file('wordcloud.png')   # 워드 클라우드 저장시 사용

    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.imshow(cloud)
    plt.show()

    return print(keyword[:10])



if __name__ == "__main__":
    # 데이터는 하나의 요약 문장 형식 
	data = "./test.txt"  
	dicpath = r'C:/mecab/mecab-ko-dic'
	keyword(data,dicpath)