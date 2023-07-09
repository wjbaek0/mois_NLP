from konlpy.tag import Mecab
import os

## 텍스트 파일의 형태소를 분석하는 파일 입니다.
## 형태소를 ' '로 구분합니다. 수정 -> line38 
## https://uwgdqo.tistory.com/363 설치법 + pip install konlpy


tokenizer  = Mecab(dicpath=r"C:\\mecab\\mecab-ko-dic") 

# input , output 경로 (txt가 있는 폴더 내부서 실행)
input_dir = os.path.join(os.getcwd())
output_dir = os.path.join(os.getcwd(), 'mecab')



def file_search(input_dir):
	file_list = [] # 파일이름 임시변수
	path_list = [] # 패스 임시변수
	for (root, directories, files) in os.walk(input_dir):
		for file in files:
			if '.txt' in file:
				file_path = os.path.join(root, file)
				path_list.append(file_path)
				ori_file_path = file_path.split('\\')[-1]
				file_list.append(ori_file_path[:-4])
	return path_list, file_list


def classification(file_path_list,file_name_list) :   
	for i,path in enumerate(file_path_list): # 패스의 모든 파일 
		text = open(path, "r",encoding='UTF8') # 텍스트파일 읽어오기
		lines = text.readlines() 
		name = "mecab_" + file_name_list[i] + ".txt" # 변환될 파일 이름 지정

		token_fin = ''
		for line in lines: # 라인별 토큰화
			token1 = tokenizer.morphs(line) 
			for k in token1:
				if len(k) >0:
					token_fin = token_fin+" "+ k
		
		mecab_save = open(output_dir+'\\'+name,'w',encoding='UTF-8')
		mecab_save.write(token_fin)

   

if __name__=='__main__':
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
  
	file_path_list, file_name_list = file_search(input_dir)
	print("전체 파일개수 : ", len(file_path_list))
	
	# mecab 분류 시작 
	classification(file_path_list, file_name_list)
 

