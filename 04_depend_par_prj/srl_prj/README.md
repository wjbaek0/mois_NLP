# SRL Project

---
### 1. SRL?
- Semantic Role Labeling : 각 서술어의 의미와 그 논항들의 의미역을 결정해 누가, 무엇을, 어떻게, 왜 등의 의미 관계를 찾아내는 과정
- 격틀사전 기반
  - 격틀(Frame); 서술어와 논항에 대한 문법 관계를 기술한 것
  - 격틀사전을 바탕으로 서술어-논항 관계에 부합하는 격틀을 선택해 의미역을 결정
  - 입력문장과 격틀 사이의 유사도 계산을 통해 의미역 결정
- 말뭉치 기반
  - 의미역이 태깅된 말뭉치를 바탕으로 기계학습 방법으로 의미역 결정
  - 데이터셋
    - [PropBank(Proposition Bank)](http://verbs.colorado.edu/~mpalmer/projects/ace.html)
    - [울산대 자연어처리연구실 UCorpus](http://nlplab.ulsan.ac.kr/doku.php?id=ucorpus)



### 2. 구현사례

- [SNU BERT-ko-srl](https://github.com/machinereading/BERT_for_Korean_SRL)

- [ETRI 언어분석 api](https://aiopen.etri.re.kr/guide_wiseNLU.php)

- [창원대-Naber NLP challenge](http://air.changwon.ac.kr/?page_id=14)

- [kakaobrain Pororo](https://github.com/kakaobrain/pororo), [Pororo SRL document](https://kakaobrain.github.io/pororo/tagging/srl.html)

  

