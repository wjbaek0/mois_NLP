import os
from konlpy.tag import Mecab

mecab = Mecab('C:\mecab\mecab-ko-dic')

# BASE_PATH = 'working'
DATA_PATH = 'working/_data'


class SentenceScoring:
    def __init__(self, file):  # file: text file directory
        self.file = file
        self.sentences = self.open_file()
        self.sentences_nn, self.all_words = self.clean_words()
        self.freq_table = self.create_word_frequency_table()
        self.sent_value = self.create_sentence_score_table()
        self.average = self.find_average_score()
        self.summary = self.generate_summary()
        self.len_summary = len(self.summary['id'])

    def open_file(self):
        with open(self.file, encoding='utf-8') as f:
            sentences = f.readlines()
        return sentences

    # def komoran_tokenizer(sent: str):
    #     words = komoran.pos(sent, join=True)
    #     words = [w.split('/')[0]
    #              for w in words if ('/NN' in w or '/VA' in w or '/VV' in w)]
    #     return words

    def clean_words(self):
        sentences_nn = []
        all_words = []

        for sent in self.sentences:
            words = mecab.pos(sent, join=True)
            words = [w.split('/')[0] for w in words if ('/NN' in w)]
            sentences_nn.append(words)

        for sent in sentences_nn:
            for word in sent:
                all_words.append(word)

        return sentences_nn, all_words

    def create_word_frequency_table(self) -> dict:
        freq_table = dict()
        for word in self.all_words:
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        freq_table = dict(
            sorted(freq_table.items(), key=lambda x: x[1], reverse=True))
        return freq_table

    def create_sentence_score_table(self) -> dict:
        sent_value = dict()
        for i in range(len(self.sentences)):
            for word, freq in self.freq_table.items():
                if word in self.sentences_nn[i]:
                    if i in sent_value:
                        sent_value[i] += freq
                    else:
                        sent_value[i] = freq
        sent_value = dict(
            sorted(sent_value.items(), key=lambda x: x[1], reverse=True))
        return sent_value

    def find_average_score(self) -> int:
        sum_values = 0
        for sentence in self.sent_value:
            sum_values += self.sent_value[sentence]

        average = int(sum_values / len(self.sent_value))
        # average = int(sum_values / len(self.sent_value)) * 1.5
        return average

    def generate_summary(self) -> str:
        summary = {}
        summary['text'] = ''
        summary['id'] = []
        summary['score'] = []
        redundancy = []

        for id in self.sent_value:
            first20 = self.sentences[id][:20]
            if first20 not in redundancy:
                summary['text'] += self.sentences[id]
                summary['id'].append(id)
                summary['score'].append(self.sent_value[id])
                redundancy.append(first20)
            if len(summary['id']) >= len(self.sentences) * 0.1:
                break
        return summary

    def print_summary(self):
        summary = self.summary
        print('======== Summary ========', summary['text'])

        print('\n======== Number of Sentences ========')
        print(
            f'Summarized: {self.len_summary} sentences / Original: {len(self.sentences)} sentences\nAverage Score: {self.average}')


if __name__ == "__main__":
    file = input('확장자(.txt)를 제외한 파일명: ')
    file += '.txt'
    SentenceScoring(f'{DATA_PATH}/{file}').print_summary()
    # print(SentenceScoring(f'{BASE_PATH}/{file}').summary)

    # SentenceScoring(file).print_summary()
    # 모든 파일들을 대상으로 <원문-요약> 형식으로 이루어진 데이터 만들기
