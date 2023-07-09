import itertools
import torch
import os 
import time
import datetime
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import pandas as pd 
import numpy as np
import seaborn as sn
import copy
import matplotlib.pyplot as plt
from transformers import BertModel
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import gc

##  python file import 
from kobert.pytorch_kobert import get_kobert_model
from kobert_tokenizer import KoBERTTokenizer, predict

################ transformers
from transformers import AdamW 
from transformers.optimization import get_cosine_schedule_with_warmup

now1 = datetime.datetime.now().strftime('%y%m%d')
time1 = datetime.datetime.now().strftime('%H%M')
tensorboard_dir = os.path.join('tensorboard', now1+'_'+time1)


################ Setting parameters
device = torch.device('cuda:3')
max_len = 300 # kobert base <= 300 이상 allocate memory 문제 , code Test시 64로 부여
batch_size = 16
num_epochs = 100  # epochs
warmup_ratio = 0.1
max_grad_norm = 5
log_interval = 200 # log 간격
learning_rate =  5e-5  # 1e-4 ~ 5e-5                                                                                                                                                                                            
drop_rate = 0.5 # drop out rate NLP => 0.5 ~ 0.8


class Config :
	now = datetime.datetime.now().strftime('%y%m%d')
	time = datetime.datetime.now().strftime('%H%M')
	txt_log_dir = os.path.join(os.getcwd(), 'log')
	LOGGING_FILE = os.path.join(txt_log_dir , now+'_'+time+'_kobert_classified_log.txt')
	model_dir = os.path.join(os.getcwd(), 'model')
	MODEL_PATH =  os.path.join(model_dir , now+'_'+time)


class BERTDataset(Dataset):  # dataset to bert_tokenizer
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset] ## SENTENCES
        self.labels = [np.int32(i[label_idx]) for i in dataset] ## LABAL

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         
    def __len__(self):
        return (len(self.labels))
    
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768, ### 고정
                 num_classes=2,   ### 클래스 수
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        
        ##### layer setting #############
        # self.classifier = nn.Sequential(
		# 	nn.Linear(hidden_size , 256),
		# 	# torch.nn.BatchNorm1d(256),
		# 	# nn.ReLU(),
		# 	nn.Linear(256, num_classes),
		# )
		#################################
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
        self.lstm = nn.LSTM(input_size=768, hidden_size=32, num_layers = 1, batch_first=True).to(device)
        self.tanh = nn.Tanh()
        self.line = nn.Linear(max_len * 32, 32).to(device)
        self.line2 = nn.Linear(32, 2).to(device)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        doc_tok, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        # h_0, c_0 = torch.randn(1, 64, 1).to(device), torch.randn(1, 64, 1).to(device)
        if self.dr_rate:
            out = self.dropout(doc_tok)
        output, (hn, cn) = self.lstm(out)
        # output_ = torch.stack(tuple([(o[-1]) for o in output]))
        output1 = output.contiguous().view([doc_tok.shape[0], -1])
        output1 = self.tanh(output1)
        output2 = self.line(output1)
        output3 = self.line2(output2)
        output3 = self.softmax(output3)
        
            
        # return self.classifier(out) ### logits
        return output3



# 학습코드 
def train(train_dataloader,valid_dataloader) :
# 모델 로그 저장을 위한 로깅 파일 호출 
	import nvidia_smi
	nvidia_smi.nvmlInit()
	with open(Config.LOGGING_FILE, 'a+t') as log:
		log.write("Training start..\n")
	
		start = time.time() # epochs 시간 체크 
		all_losses = []
		all_acc = []
		all_epochs = []
  
		for e in range(num_epochs):
			print('-' * 100)
			print("Epoch {}/{}\n".format(e, num_epochs-1))
			log.write('-' * 100)
			log.write("\nEpoch {}/{}\n".format(e, num_epochs-1))
			ep_start = time.time()
			train_acc = 0.0
			val_acc = 0.0
			model.train()

			for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
				optimizer.zero_grad()
				token_ids = token_ids.long().to(device)
				segment_ids = segment_ids.long().to(device)
				valid_length= valid_length
				label = label.long().to(device)
				out = model(token_ids, valid_length, segment_ids)
				
				loss = loss_fn(out, label)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				train_acc += calc_accuracy(out, label)
	
				if batch_id % log_interval == 0:
					f_loss = float(loss.data.cpu().numpy())
					print("epoch {} loss {} ".format(e+1, f_loss))
					log.write("epoch {} loss {} ".format(e+1, f_loss))
					all_losses.append(f_loss)
     
			handle = nvidia_smi.nvmlDeviceGetHandleByIndex(3)
			info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
			print("Train: Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(3, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
			print(" train acc {}".format( round((train_acc / (batch_id+1)),4)))
			log.write(" train acc {}".format( round((train_acc / (batch_id+1)),4)))
			all_acc.append(train_acc / (batch_id+1))
   
		
			## 모델 추론 = 현재 1 epoch당 추론중  
			model.eval()
			val_loss = 0.0
			for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(valid_dataloader):
				token_ids = token_ids.long().to(device)
				segment_ids = segment_ids.long().to(device)
				valid_length= valid_length
				label = label.long().to(device)
				out = model(token_ids, valid_length, segment_ids)
				val_loss = loss_fn(out, label)
				# val_loss.backward()
				val_acc += calc_accuracy(out, label)
			f_val_loss = float(val_loss.data.cpu().numpy())
			print("\neval --> epoch {}, val loss {}, val acc {}".format(e+1,f_val_loss, round((val_acc / (batch_id+1)),4)))
			log.write("\neval --> epoch {}, val loss {}, val acc {}".format(e+1,f_val_loss, round((val_acc / (batch_id+1)),4)))
   
			handle = nvidia_smi.nvmlDeviceGetHandleByIndex(3)
			info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
			print("Train: Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(3, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
   
			all_losses.append(f_val_loss) 
			all_acc.append(val_acc / (batch_id+1))
			all_epochs.append(e)
   
			ep_end = time.time()
			print('Time : {}'.format(datetime.timedelta(seconds=ep_end-ep_start)))
			log.write('\nTime : {}\n'.format(datetime.timedelta(seconds=ep_end-ep_start)))
   
			model_wts = copy.deepcopy(model.state_dict()) # model 추론을 위한 state_Dict 저장 
			BEST_MODEL_PATH = os.path.join(Config.MODEL_PATH, 'kobert_model_{}_lstm.pth'.format(e))
			torch.save(model_wts, BEST_MODEL_PATH) # 모델 저장
   
   
		nvidia_smi.nvmlShutdown()
		ckpt_PATH = os.path.join(Config.MODEL_PATH, 'kobert_model_DenseNet_ckpt.pt')
		print("----------------ckpt 저장-----------------")
		
		torch.save({'epoch' : all_epochs,
			# 'model_state_dict' : model.state_dict(),
			# 'optimizer_state_dict' : optimizer.state_dict(),
			'loss' : all_losses,
			'acc':all_acc},
			ckpt_PATH
		)
		# 최종 시간 체크 
		end = time.time()
		log.write('=' * 100)
		log.write('\n[Training Complete: {}]'.format(datetime.timedelta(seconds=end-start)))
		print('=' * 100)
		print('[Training Complete: {}]'.format(datetime.timedelta(seconds=end-start)))

# * 데이터셋 저장할 경로 확인 후 없다면 새로 생성
def make_dirs(): 
	if not os.path.exists(Config.txt_log_dir):
		os.makedirs(Config.txt_log_dir)  
	if not os.path.exists(tensorboard_dir):
		os.makedirs(tensorboard_dir)
	if not os.path.exists(Config.model_dir):
		os.makedirs(Config.model_dir)    
	if not os.path.exists(Config.MODEL_PATH):
		os.makedirs(Config.MODEL_PATH)
	if os.path.exists(Config.LOGGING_FILE):
		os.remove(Config.LOGGING_FILE)
		print('학습 로그 파일 초기화.')
    
# csv에서 데이터를 가져오는 함수    
def get_data(ratings_train,ratings_vaild,ratings_test):
    # .py와 동일 경로에 각각의 csv 
	train_data = pd.read_csv(ratings_train, encoding='utf8')
	val_data = pd.read_csv(ratings_vaild, encoding='utf8')
	test_data = pd.read_csv(ratings_test, encoding='utf8')
 
	# 임시 저장 데이터셋 
	dataset_train = []
	dataset_val = []
	dataset_test = []

	# 라벨 불러오기 0,1 부여
	train_data.loc[(train_data['label'] == "Construction"), 'label'] = 0  # 공사장 => 0
	train_data.loc[(train_data['label'] == "Fire"), 'label'] = 1  # 화재 => 1

	val_data.loc[(val_data['label'] == "Construction"), 'label'] = 0  # 공사장 => 0
	val_data.loc[(val_data['label'] == "Fire"), 'label'] = 1  # 화재 => 1

	test_data.loc[(test_data['label'] == "Construction"), 'label'] = 0  # 공사장 => 0
	test_data.loc[(test_data['label'] == "Fire"), 'label'] = 1  # 화재 => 1

 
	for txt, label in zip(train_data['data'], train_data['label'])  :
		data1 = []   
		data1.append(txt)
		data1.append(str(label))

		dataset_train.append(data1)
  
	for txt, label in zip(test_data['data'], test_data['label'])  :
		data2 = []   
		data2.append(txt)
		data2.append(str(label))

		dataset_test.append(data2)
  
	for txt, label in zip(val_data['data'], val_data['label'])  :
		data3 = []   
		data3.append(txt)
		data3.append(str(label))

		dataset_val.append(data3)
  
	return dataset_train, dataset_test, dataset_val

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


###################################################################################################### 추론 코드 ######
def predict(model,test_dataloader):

	model.eval()
	y_pred=[]
	y_true=[]
	confidence=[]
	cnt = 0
	
	with torch.no_grad():
		for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
			
			token_ids = token_ids.long().to(device)
			segment_ids = segment_ids.long().to(device)
			valid_length= valid_length
			label = label.long().to(device)

			# print("token_ids , valid_length , segment_ids ", token_ids,valid_length,segment_ids)
	
			out = model(token_ids, valid_length, segment_ids)
			y_true.append(np.array(label.data.cpu().numpy()).tolist())
			for i in out:
				logits = i
				softmax = nn.Softmax(dim=-1)
				confidence.append(softmax(logits))
	
	
				# 화재 결과를 대부분 공사장으로 잡아버리는 오류가 발생중. 
				########## y_pred ==> 내용에 "화재"가 있을시 화재로 잡아버리도록..  
	
				y_pred.append(torch.argmax(i).cpu().numpy().tolist())

	y_true_sort = list(itertools.chain(*y_true))
	print("정답 라벨 리스트(0:construct , 1:fire) : ", y_true_sort)
	print(">> 추론된 값은 :  " , y_pred )
	Construction_score = []
	for i in confidence :
		Construction_score.append(round((i.tolist()[0]), 5)) 
	print(">> 추론값의 confidence :  " , Construction_score )
	# print("정답 라벨 리스트(0:construct , 1:fire) : ", y_true_sort)
	# print(">> 추론된 값은 :  " , y_pred )
	# print(">> 추론값의 confidence :  " , confidence )
	for i,j in zip(y_true_sort,y_pred):
		if i == j :
			cnt += 1
	
	print("추론 정확도 >>>> : ", round((cnt/len(y_pred))*100, 2) , " %")

	classes = ('construct','fire')
	cf_matrix = confusion_matrix(y_true_sort, y_pred)
	# df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
	#                     columns = [i for i in classes]) # 비율
	df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
						columns = [i for i in classes]) # 수치 
	plt.figure(figsize = (12,7))
	sn.heatmap(df_cm, annot=True)
	plt.title('Kobert predict matrix', fontsize=15)
	plt.savefig('output.png')
	plt.show()  

###### ACC/LOSS #########################################################################################################
def ckpt_plot(checkpoint_epoch,checkpoint_loss,checkpoint_acc):

	train_loss = []
	val_loss = []
	train_acc = []
	val_acc = []
	
	print("checkpoint_epoch >>> ", len(checkpoint_epoch)) # 최종 epochs

	for index in range(len(checkpoint_loss)) :
		if index % 2 == 0 :
			train_loss.append(checkpoint_loss[index])
		else :    
			val_loss.append(checkpoint_loss[index])

	
	
	for index in range(len(checkpoint_acc)) :
		if index % 2 == 0 :
			train_acc.append(checkpoint_acc[index])
		else :    
			val_acc.append(checkpoint_acc[index])
			
	# train_loss, val_loss
	plt.figure(figsize=(10,5))
	plt.plot(train_loss,'r', label="train_loss")
	plt.plot(val_loss,'g', label="val_loss")
	plt.legend(ncol=2, loc="upper right")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.legend()
	plt.savefig('loss.png')
	plt.show()

	# train_acc, val_acc
	plt.figure(figsize=(10,5))
	plt.plot(train_acc, label="train_acc")
	plt.plot(val_acc, label="val_acc")
	plt.legend(ncol=2, loc="upper right")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend()
	plt.savefig('acc.png')
	plt.show()

    
if __name__=='__main__':
    # Data load를 위한 기본 모델, 보카 , 토크나이저 호출 
	tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
	pre_model, vocab = get_kobert_model('skt/kobert-base-v1',tokenizer.vocab_file)

	dataset_train, dataset_val , dataset_test = get_data('ratings_train.csv','ratings_valid.csv', 'ratings_test.csv') # csv 데이터 불러오기 
	tok=tokenizer.tokenize # 토크나이징 => 서브워드화 + 품사태깅
	
	data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
	data_val = BERTDataset(dataset_val, 0, 1, tok, vocab, max_len, True, False)
	data_test = BERTDataset(dataset_test,0, 1, tok, vocab,  max_len, True, False)
	train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=4) # cpu = 0
	valid_dataloader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, num_workers=4) # cpu = 0
	test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=4) # cpu = 0

	print("Train , valid , test 갯수 : " , len(dataset_train),len(dataset_val),len(dataset_test))
	
	Task = int(input("작업할 Task 숫자입력 ->(1. Train , 2. Predict , 3. Acc/Loss  : " ))
	
 
	if Task == 1 :
		make_dirs()
		# BERT 모델 불러오기
		model = BERTClassifier(pre_model, dr_rate=drop_rate).to(device)
	
		########### 옵티마이져 , loss_fuction, step 설정 
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [ # 옵티마이저 파라미터 세팅 
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
		loss_fn = nn.CrossEntropyLoss() # loss func
		t_total = len(train_dataloader) * num_epochs
		warmup_step = int(t_total * warmup_ratio)
		scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
		#################################################
	
		######### 학습함수 호출 ##############
		train(train_dataloader,valid_dataloader) 
 
	elif Task == 2 :
		######## 메모리 최적화를 위한 코드##########################
		gc.collect()
		torch.cuda.empty_cache()
		##########################################################
	
		model = BERTClassifier(pre_model, dr_rate=drop_rate).to(device)
		model.load_state_dict(torch.load('./kobert_model_1_lstm.pth'), strict=False)
	
		####### 추론 함수 호출 ###############
		predict(model,test_dataloader)
 

	elif Task == 3 :
		############# Loss , Acc plot 출력을 위한, 체크포인트 모델 불러오기####################################
	
		ckpt_model = BertModel.from_pretrained("skt/kobert-base-v1", return_dict=False)
		checkpoint = torch.load('./kobert_model_DenseNet_ckpt.pt') # ckpt 경로 확인
		checkpoint_epoch = checkpoint["epoch"]
		checkpoint_loss = checkpoint["loss"]
		checkpoint_acc = checkpoint["acc"]
		# 체크포인트 출력 함수 호출하기 ############## 
		ckpt_plot(checkpoint_epoch,checkpoint_loss,checkpoint_acc)
		#####################################################################################################
