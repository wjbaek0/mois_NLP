from konlpy.tag import Mecab
import os

#품사값이 명사인것만 추출 -> 수정 : line44
## https://uwgdqo.tistory.com/363 설치법 + pip install konlpy
tokenizer  = Mecab(dicpath=r"C:\\mecab\\mecab-ko-dic") 

# input , output 경로 (txt가 있는 폴더 내부서 실행)
input_dir = os.path.join(os.getcwd(), 'mecab')
output_dir = os.path.join(os.getcwd(), 'mecab_pos')



def file_search(input_dir):
	file_list = [] # 파일이름 임시변수
	path_list = [] # 패스 임시변수
	for (root, _, files) in os.walk(input_dir):
		for file in files:
			if '.txt' in file:
				file_path = os.path.join(root, file)
				path_list.append(file_path)
				ori_file_path = file_path.split('\\')[-1]
				file_list.append(ori_file_path[:-4])
	return path_list, file_list


def classification(file_path_list,file_name_list) :   
	for i,path in enumerate(file_path_list): # 패스의 모든 파일 
		lines = [] # text lines
		mecab_dic = {} # mecab dic
		NNG_list = [] # NNG list 
		global count
		count = 1   
  
		text = open(path, "r",encoding='UTF8') # 텍스트파일 읽어오기
		lines = text.readlines() 
		name = "pos_" + file_name_list[i] # 변환될 파일 이름 지정 
        
		out = tokenizer.pos(lines[0])
		for i in out : #  key , value 분리하여 하나의 사전에 등록
			mecab_dic[i[0]] = i[1] # 품사태깅 리스트 
    
		for key, value in mecab_dic.items():
			if value == 'NNG' or value == 'NNP': # 품사값이 명사인것만 추출 NNG , NNP를 조절 
				NNG_list.append(key) # 명사 리스트 

		# 저장될 텍스트파일 생성  
		mecab_save = open(output_dir+'\\'+name+".txt",'w',encoding='UTF-8')
		for no,NNG_val in enumerate(NNG_list) :
			mecab_save.write(NNG_val)
			mecab_save.write(' ')
   
			if no == 509  :  # 1번 파일이 510개가 차는 시점 
				while 1: # 최종데이터까지 반복 
					n = count * 510 # 토큰 임계점 시작 
					next_n = n + 510 # 다음 토큰 임계점
		
					if index_exists(NNG_list, next_n) : # 임계점 인덱스가 있는지 확인
						NNG_over_list = NNG_list[n:next_n] 
					else :
						NNG_over_list = NNG_list[n:] # 최종데이터가 다음 임계점 이전

					count = count + 1
					if index_exists(NNG_list, n) : # 시작할 인덱스가 있는지 없는지 확인
						pass
					else :
						break
					mecab_over_save = open(output_dir+'\\'+name+'_'+str(count)+".txt",'w',encoding='UTF-8')
     
					for val in NNG_over_list:
						mecab_over_save.write(val)
						mecab_over_save.write(' ')
      
				break	   
   
def index_exists(arr, i):
    return (0 <= i < len(arr)) or (-len(arr) <= i < 0) 
   
   
   

if __name__=='__main__':
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
  
	file_path_list, file_name_list = file_search(input_dir)
	print("전체 파일개수 : ", len(file_path_list))
	
	# mecab 분류 시작 
	classification(file_path_list, file_name_list)
 

