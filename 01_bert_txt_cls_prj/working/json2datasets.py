import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
# from pykospacing import Spacing



### 각 폴더에 - TEXT - > DATASETS
def text(textdir,datasets_dir) :

	
 #### JSON 처리부분
	# with open(jsondir,'r', encoding='utf-8') as f: # json 불러오기
	# 	json_df = json.load(f)
	# 	df = pd.DataFrame(json_df)
	# 	spacing = Spacing()
	# 	for idx,val in enumerate(df):   
	# 		title = datasets_dir + "\\" + val
	# 		f = open(title,'w', encoding='utf-8')
	# 		for i in df[val]["text"] :
	# 			i = spacing(i) # spacing 적용 (띄어쓰기 재정렬) 
	# 			f.write(i+'\n')
	# 		f.close()
   
## TEXT 데이터셋 만드는 부분  ####
		#### 290 토큰을 맞추기 위한 코드 (사용시 주석 해제)
		# 	if len(result[idx]) > 290 : 
		# 		new_list = list_chunk(result[idx], 290) # 290개씩 분할 -> list 
		# 		for co in new_list :
		# 			new_data.append(co) # 나눠진 TEXT로 append
		# 			new_label.append(df['label'][idx]) # 같은 라벨로 append
     
		# 	else : # 290보다 길이가 짧은 경우 
	공사장_list = os.listdir(os.path.join(textdir,"사업장대규모인적사고"))
	화재_list = os.listdir(os.path.join(textdir,"다중밀집시설대형화재"))
	# 6 class 
	철도교통사고_list = os.listdir(os.path.join(textdir,"철도교통사고"))
	해양선박사고_list = os.listdir(os.path.join(textdir,"해양선박사고"))
	항공기사고_list = os.listdir(os.path.join(textdir,"항공기사고"))
	도로교통사고_list = os.listdir(os.path.join(textdir,"도로교통사고"))
	유해화학물질사고_list = os.listdir(os.path.join(textdir,"유해화학물질사고"))
	감염병_list = os.listdir(os.path.join(textdir,"감염병"))
	
     
    # 토큰화에 사용될 리스트
	F_data = [] 
	F_label = []
 
	C_data = [] 
	C_label = []
	
	all_data1 = []
	all_label1 = []
 
	all_data2 = []
	all_label2 = []
 
	all_data3 = []
	all_label3 = []
 
	all_data4 = []
	all_label4 = []
 
	all_data5 = []
	all_label5 = []
 
	all_data6 = []
	all_label6 = []
###################################################################################  
	for i in 공사장_list :
		i_dir = os.path.join(textdir,"사업장대규모인적사고", i)
		with open(i_dir,'r',encoding='utf-8') as f:
			txt1 = f.read()
			C_data.append(txt1)
			C_label.append("Construction")

	for j in 화재_list :
		j_dir = os.path.join(textdir,"다중밀집시설대형화재", j)
		with open(j_dir,'r',encoding='utf-8') as f:
			txt2 = f.read()
			F_data.append(txt2)
			F_label.append("Fire")
   
	for i in 철도교통사고_list :
		i_dir = os.path.join(textdir,"철도교통사고", i)
		with open(i_dir,'r',encoding='utf-8') as f:
			txt3 = f.read()
			all_data1.append(txt3)
			all_label1.append("train")

	for j in 해양선박사고_list :
		j_dir = os.path.join(textdir,"해양선박사고", j)
		with open(j_dir,'r',encoding='utf-8') as f:
			txt4 = f.read()
			all_data2.append(txt4)
			all_label2.append("ship")
	for i in 항공기사고_list :
		i_dir = os.path.join(textdir,"항공기사고", i)
		with open(i_dir,'r',encoding='utf-8') as f:
			txt5 = f.read()
			all_data3.append(txt5)
			all_label3.append("aircraft")

	for j in 도로교통사고_list :
		j_dir = os.path.join(textdir,"도로교통사고", j)
		with open(j_dir,'r',encoding='utf-8-sig') as f:
			txt6 = f.read()
			all_data4.append(txt6)
			all_label4.append("road")
	for i in 유해화학물질사고_list :
		i_dir = os.path.join(textdir,"유해화학물질사고", i)
		with open(i_dir,'r',encoding='utf-8') as f:
			txt7 = f.read()
			all_data5.append(txt7)
			all_label5.append("chemical")

	for j in 감염병_list :
		j_dir = os.path.join(textdir,"감염병", j)
		with open(j_dir,'r',encoding='utf-8') as f:
			txt8 = f.read()
			all_data6.append(txt8)
			all_label6.append("infect")
###############################################################################
 
	df_Construction = pd.DataFrame({'label':C_label, 'data':C_data}) # 분할된 리스트 

	# 공사장 정렬
	df_Con = df_Construction[['label','data']]
	df_Con.loc[:,'data'] = df_Con.loc[:,'data'].apply(lambda x : x.replace('○','')) ##  데이터상의 ○를 제거 
	df_Con.loc[:,'data'] = df_Con.loc[:,'data'].apply(lambda x : x.replace('\n',' ')) ## 일직선화 
	df_Con.loc[:,'label'] = df_Con.loc[:,'label'].apply(lambda x : x)

	Con_dataset_train, Con_dataset_another = train_test_split(df_Con, test_size=0.2, random_state=0) 
	Con_dataset_val , Con_dataset_test = train_test_split(Con_dataset_another, test_size=0.5, random_state=0)
 
	df_fire = pd.DataFrame({'label':F_label, 'data':F_data}) # 분할된 리스트 

	# 화재 정렬
	df_Fi = df_fire[['label','data']]
	df_Fi.loc[:,'data'] = df_Fi.loc[:,'data'].apply(lambda x : x.replace('\n',' '))
	df_Fi.loc[:,'label'] = df_Fi.loc[:,'label'].apply(lambda x : x)

	Fire_dataset_train, Fire_dataset_another = train_test_split(df_Fi, test_size=0.2, random_state=0) 
	Fire_dataset_val , Fire_dataset_test = train_test_split(Fire_dataset_another, test_size=0.5, random_state=0)

###############################################################################################

	df_all_data1 = pd.DataFrame({'label':all_label1, 'data':all_data1}) # 분할된 리스트 
	# 사업장대규모인적사고
	df_data1 = df_all_data1[['label','data']]
	df_data1.loc[:,'data'] = df_data1.loc[:,'data'].apply(lambda x : x.replace('○','')) ##  데이터상의 ○를 제거 
	df_data1.loc[:,'data'] = df_data1.loc[:,'data'].apply(lambda x : x.replace('\n',' ')) ## 일직선화 
	df_data1.loc[:,'label'] = df_data1.loc[:,'label'].apply(lambda x : x)

	D1_dataset_train, D1_dataset_another = train_test_split(df_data1, test_size=0.2, random_state=0) 
	D1_dataset_val , D1_dataset_test = train_test_split(D1_dataset_another, test_size=0.5, random_state=0)

	df_all_data2 = pd.DataFrame({'label':all_label2, 'data':all_data2}) # 분할된 리스트 
	# 공사장 정렬
	df_data2 = df_all_data2[['label','data']]
	df_data2.loc[:,'data'] = df_data2.loc[:,'data'].apply(lambda x : x.replace('○','')) ##  데이터상의 ○를 제거 
	df_data2.loc[:,'data'] = df_data2.loc[:,'data'].apply(lambda x : x.replace('\n',' ')) ## 일직선화 
	df_data2.loc[:,'label'] = df_data2.loc[:,'label'].apply(lambda x : x)

	D2_dataset_train, D2_dataset_another = train_test_split(df_data2, test_size=0.2, random_state=0) 
	D2_dataset_val , D2_dataset_test = train_test_split(D2_dataset_another, test_size=0.5, random_state=0)


	# df_all_data3 = pd.DataFrame({'label':all_label3, 'data':all_data3}) # 분할된 리스트 
	# # 공사장 정렬
	# df_data3 = df_all_data3[['label','data']]
	# df_data3.loc[:,'data'] = df_data3.loc[:,'data'].apply(lambda x : x.replace('○','')) ##  데이터상의 ○를 제거 
	# df_data3.loc[:,'data'] = df_data3.loc[:,'data'].apply(lambda x : x.replace('\n',' ')) ## 일직선화 
	# df_data3.loc[:,'label'] = df_data3.loc[:,'label'].apply(lambda x : x)

	# D3_dataset_train, D3_dataset_another = train_test_split(df_data3, test_size=0.2, random_state=0) 
	# D3_dataset_val , D3_dataset_test = train_test_split(D3_dataset_another, test_size=0.5, random_state=0)



	# df_all_data4 = pd.DataFrame({'label':all_label4, 'data':all_data4}) # 분할된 리스트 
	# # 공사장 정렬
	# df_data4 = df_all_data4[['label','data']]
	# df_data4.loc[:,'data'] = df_data4.loc[:,'data'].apply(lambda x : x.replace('○','')) ##  데이터상의 ○를 제거 
	# df_data4.loc[:,'data'] = df_data4.loc[:,'data'].apply(lambda x : x.replace('\n',' ')) ## 일직선화 
	# df_data4.loc[:,'label'] = df_data4.loc[:,'label'].apply(lambda x : x)

	# D4_dataset_train, D4_dataset_another = train_test_split(df_data4, test_size=0.2, random_state=0) 
	# D4_dataset_val , D4_dataset_test = train_test_split(D4_dataset_another, test_size=0.5, random_state=0)

	df_all_data5 = pd.DataFrame({'label':all_label5, 'data':all_data5}) # 분할된 리스트 
	# 공사장 정렬
	df_data5 = df_all_data5[['label','data']]
	df_data5.loc[:,'data'] = df_data5.loc[:,'data'].apply(lambda x : x.replace('○','')) ##  데이터상의 ○를 제거 
	df_data5.loc[:,'data'] = df_data5.loc[:,'data'].apply(lambda x : x.replace('\n',' ')) ## 일직선화 
	df_data5.loc[:,'label'] = df_data5.loc[:,'label'].apply(lambda x : x)

	D5_dataset_train, D5_dataset_another = train_test_split(df_data5, test_size=0.2, random_state=0) 
	D5_dataset_val , D5_dataset_test = train_test_split(D5_dataset_another, test_size=0.5, random_state=0)

	# df_all_data6 = pd.DataFrame({'label':all_label6, 'data':all_data6}) # 분할된 리스트 
	# # 공사장 정렬
	# df_data6 = df_all_data6[['label','data']]
	# df_data6.loc[:,'data'] = df_data6.loc[:,'data'].apply(lambda x : x.replace('○','')) ##  데이터상의 ○를 제거 
	# df_data6.loc[:,'data'] = df_data6.loc[:,'data'].apply(lambda x : x.replace('\n',' ')) ## 일직선화 
	# df_data6.loc[:,'label'] = df_data6.loc[:,'label'].apply(lambda x : x)

	# D6_dataset_train, D6_dataset_another = train_test_split(df_data6, test_size=0.2, random_state=0) 
	# D6_dataset_val , D6_dataset_test = train_test_split(D6_dataset_another, test_size=0.5, random_state=0)

 
 ###############################################################
	# 데이터셋 통합 
	# dataset_train = pd.concat([Con_dataset_train,Fire_dataset_train,D1_dataset_train , D2_dataset_train,D3_dataset_train,D4_dataset_train,D5_dataset_train,D6_dataset_train])
	dataset_train = pd.concat([Con_dataset_train,Fire_dataset_train,D1_dataset_train , D2_dataset_train,D5_dataset_train])
	# dataset_val = pd.concat([Con_dataset_val,Fire_dataset_val,D1_dataset_val , D2_dataset_val,D3_dataset_val,D4_dataset_val,D5_dataset_val,D6_dataset_val])
	dataset_val = pd.concat([Con_dataset_val,Fire_dataset_val,D1_dataset_val , D2_dataset_val,D5_dataset_val,])
	# dataset_test = pd.concat([Con_dataset_test,Fire_dataset_test,D1_dataset_test , D2_dataset_test,D3_dataset_test,D4_dataset_test,D5_dataset_test,D6_dataset_test])
	dataset_test = pd.concat([Con_dataset_test,Fire_dataset_test,D1_dataset_test , D2_dataset_test,D5_dataset_test])
	
	train_txt_dir = os.path.join(datasets_dir,'ratings_train.csv') ### 변경
	val_txt_dir = os.path.join(datasets_dir,'ratings_val.csv') ### 변경 
	test_txt_dir = os.path.join(datasets_dir,'ratings_test.csv') ### 변경 
	
	dataset_train.to_csv(train_txt_dir, encoding = 'utf-8-sig' , index=False)
	dataset_val.to_csv(val_txt_dir, encoding = 'utf-8-sig' , index=False)
	dataset_test.to_csv(test_txt_dir, encoding = 'utf-8-sig' , index=False)  

	# df_d.to_csv(train or val or test, sep=',', encoding= 'utf-8', header=['label','text'], index=False)
	print("OK.")

def list_chunk(lst, n): # TEXT 분할하는 코드
    return [lst[i:i+n] for i in range(0, len(lst), n)] 

if __name__ == "__main__":
	# jsondir = os.path.join(os.getcwd(),'sum_lexrank_27.json') # 요약된 json 파일

	# textdir = os.path.join(os.getcwd(),'미래아이티_추가 클래스 6종')
	textdir = os.path.join(os.getcwd(),'data')
	print("===============================",textdir)
	# datasets_dir = os.path.join(os.getcwd(),'미래아이티_추가 클래스 6종','datasets') # 저장될 save 폴더
	datasets_dir = os.path.join(os.getcwd(),'output_data') # 저장될 save 폴더
	
 
	text(textdir,datasets_dir)
