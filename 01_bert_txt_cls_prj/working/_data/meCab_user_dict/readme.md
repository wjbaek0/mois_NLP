
# nnp.csv : 고유명사 사전
# person.csv : 인명 사전
# place.csv : 지명 사전

[형식]
단어, 0, 0, 우선순위, 품사 태그, 종성 유무, 읽기, 타입, 첫 번째 품사, 마지막 품사, 원형, 인덱스 표현

meCab의 사용자 사전에 고유명사를 입력. add 하는 코드



1. C:\mecab\user-dic 경로에 추가된 사용자 nnp.csv를 

2. mecab\mecab-ko-dic경로 에 파워 쉘 명령어를 통하여 등록

3. .\tools\add-userdic-win.ps1 파일 실행 (win 사전등록)

! Set-ExecutionPolicy Unrestricted 권한오류 발생시 입력

