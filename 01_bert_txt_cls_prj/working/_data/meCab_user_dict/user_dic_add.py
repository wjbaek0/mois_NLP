
def dic_search() : 
	# 사전 내용 확인 
	with open("C:/mecab/user-dic/nnp.csv", 'r', encoding='cp949') as f: 
		file_data = f.readlines()
	return file_data

def dic_add() :
    
	# 사전 내용 확인 
	with open("C:/mecab/user-dic/nnp.csv", 'r', encoding='cp949') as f: 
		file_data = f.readlines()
	print("사전내용 확인 \n",file_data)
	# 사전 추가 
	while(1):
		input_data = input('추가할 명사 입력 (종료는 exit): ')
		if input_data == 'exit':
			break
		file_data.append(f'{str(input_data)},,,,NNP,*,F,{str(input_data)},*,*,*,*,*\n')

 
	print("추가 데이터 확인 \n ",file_data)
	with open("C:/mecab/user-dic/nnp.csv", 'w', encoding='cp949') as f: 
		for line in file_data: 
			f.write(line)
	print("추가 완료.")

def dic_rank() : # 우선순위 0으로 변경 - 추가한 사전이 결과에 적용되지 않을때 우선순위 변경이 필요 
    
	with open("C:/mecab/mecab-ko-dic/user-nnp.csv", 'r', encoding='cp949') as f: 
		file_data = f.readlines()
  
	for data in file_data :
		print(data)
		data[3]='0'# 각 행에서 3번째 순위 인덱스를 0번째로 변경(최우선)

	with open("C:/mecab/mecab-ko-dic/user-nnp.csv", 'w', encoding='cp949') as f: 
		for line in file_data: 
			f.write(line)

	print("우선순위 변경 완료 ") 

if __name__ == "__main__":
    
    dic_add() # 사전추가 
    # dic_search()
    # 파워쉘 c\mecab --> .\tools\add-userdic-win.ps1 실행 > Set-ExecutionPolicy Unrestricted 권한오류시 입력 
    # dic_rank()