import os
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer

mecab = Mecab('C:\mecab\mecab-ko-dic')

# BASE_PATH = 'working'
DATA_PATH = 'working/_data'


class TfidfScoring:
    def __init__(self, file):  # file: text file directory
        self.file = file
        self.sentences = self.open_file()
        self.sentences_nn = self.clean_words()
        self.sent_value = self.create_word_tfidf_table()
        self.average = self.find_average_score()
        self.summary = self.generate_summary()
        self.len_summary = len(self.summary['id'])

    def open_file(self):
        with open(self.file, encoding='utf-8') as f:
            sentences = f.readlines()
        return sentences

    def clean_words(self):
        def mecab_tokenizer(sent: str):
            words = mecab.pos(sent, join=True)
            words = [w.split('/')[0]
                     for w in words if ('/NN' in w)]
            return words
        sentences_nn = []
        for sent in self.sentences:
            words = ' '.join(mecab_tokenizer(sent))
            sentences_nn.append(words)
        return sentences_nn

    def create_word_tfidf_table(self) -> dict:
        tfidf = TfidfVectorizer()
        tfidf_array = tfidf.fit_transform(self.sentences_nn).toarray()
        sent_value = {}
        for i in range(len(tfidf_array)):
            sent_value[i] = tfidf_array[i].sum()
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
        summaries = {}
        summaries['text'] = ''
        summaries['id'] = []
        summaries['score'] = []
        redundancy = []

        for id in self.sent_value:
            first20 = self.sentences[id][:20]
            if first20 not in redundancy:
                summaries['text'] += self.sentences[id]
                summaries['id'].append(id)
                summaries['score'].append(self.sent_value[id])
                redundancy.append(first20)
            if len(summaries['id']) >= len(self.sentences) * 0.1:
                break
        return summaries

    def print_summary(self):
        summary = self.summary
        print('======== Summary ========', summary['text'])

        print('\n======== Number of Sentences ========')
        print(
            f'Summarized: {self.len_summary} sentences / Original: {len(self.sentences)} sentences')


if __name__ == "__main__":
    file = input('확장자(.txt)를 제외한 파일명: ')
    file += '.txt'
    TfidfScoring(f'{DATA_PATH}/{file}').print_summary()
