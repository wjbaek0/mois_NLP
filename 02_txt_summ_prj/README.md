# Text summarization Project

### 요약 방식
3가지 방식으로 테스트하여 최종적으로 LexRank 방식의 요약을 선택

|파일|설명|채택|
|-|:-|:-:|
|sentence_scoring.py|Sentence Scoring 방식 요약
|tfidf_scoring.py|TF-IDF로 문장 점수 부여
|lex_ranking.py|TextRank와 유사한 LexRank 방법으로 요약|O|

[사용법(공통)]<br/>입력으로 내부에 지정해둔 경로에 존재하는 하나의 파일명(확장자 제외)을 받아,<br/>결과로 요약 추출된 문장들을 출력 (저장 X)

```python
python lex_ranking.py
```

<br/>

### 저장

|파일|설명|
|-|:-|
|make_json.py|위의 lex_ranking.py를 활용해 각 파일을 요약 후 하나의 JSON 파일에 저장

[사용법]<br/>요약할 TXT 파일들이 모여있는 경로를 소스 내부에 지정하고,<br/>마찬가지로 JSON파일이 저장될 경로도 지정하여 스크립트 실행

```python
python make_json.py
```

<br/>

### lexrankr
##### 한국어에 적합하도록 구현된 LexRank 패키지
```linux
pip install lexrankr
```
위 명령어로 설치된 패키지 폴더에 인덱스 추출, 누락 방지 등의 후처리를 위해 수정한 lexrankr.py를 덮어쓰기

(수정된 lexrank.py 파일 링크<br/>&nbsp;https://github.com/XAI-AI-LAB/004_mois/blob/55123f3afb6b8f8207fabfb6a612b23c913577e9/02_txt_summ_prj/working/_git_src/lexrankr.py)
