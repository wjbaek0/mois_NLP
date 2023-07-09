import torch
import os , sys, logging as log
import random
import time
import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from kobert import get_kobert_model
from kobert_tokenizer import KoBERTTokenizer

now1 = datetime.datetime.now().strftime('%y%m%d')
time1 = datetime.datetime.now().strftime('%H%M')

device = torch.device('cuda:0')

# Setting parameters
max_len = 300 # kobert base <= 300 학습시 allocate memory 문제
batch_size = 8
drop_rate = 0.7 # drop out rate NLP => 0.5 ~ 0.8
num_classes = 8 # 학습할 데이터셋, 모델과 추론할 데이터의 클래스 갯수와 인덱스는 항상 같아야함 


class BERTDataset(Dataset):  # dataset to bert_tokenizer
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         

    def __len__(self):
        return (len(self.labels))
        
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=num_classes,   ###   클래스 수 조정 
                 dr_rate=0,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size , 256),
			# torch.nn.BatchNorm1d(256),
            # nn.ReLU(),
			nn.Linear(256, num_classes),
		)
             
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
            
        return self.classifier(out) ### logits 

def Classificaion_predict(file):
	log = ""
	pred = ""
	bardata = []

	with open("./classification_log.txt", 'w+t') as log:
		start = time.time() # 시간 체크 
		print("<classification predict start....>")
		log.write("<classification predict start....>\n")
		##모델 호출
		tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
		model, vocab = get_kobert_model('skt/kobert-base-v1',tokenizer.vocab_file)
  
		model.load_state_dict(torch.load('/home/ubuntu/workspace/nlp_integrate/Classification/kobert_model_20.pth'), strict=False) # weight 
		model = BERTClassifier(model, dr_rate=drop_rate).to(device)
		tok=tokenizer.tokenize 
	
		model.eval()
		y_pred = []
		confidence=[] 	
		## file 경로 "절대지정"으로 바꿔줌 #!!!!!!!!!!!!!!!!!!
		file_dir = '/home/ubuntu/workspace/nlp_integrate/out_puts/'+file
		with open(file_dir,'r',encoding='utf-8')as f:
			text = f.read()
	
		textset = [[text,'0']]
		text_data = BERTDataset(textset,0, 1, tok, vocab,  300, True, False)
		text_dataloader = torch.utils.data.DataLoader(text_data, batch_size=batch_size, num_workers=0)
	
		for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(text_dataloader):
			token_ids = token_ids.long().to(device)
			segment_ids = segment_ids.long().to(device)
			valid_length= valid_length

			out = model(token_ids, valid_length, segment_ids)

			for i in out:
				softmax = nn.Softmax(dim=-1)
				confidence.append(softmax(i))
				y_pred.append(torch.argmax(i).cpu().numpy().tolist())

		# Construction , fire
		if y_pred[0] == 0 :
			pred = "사업장대규모인적사고"
		elif y_pred[0] == 1 :
			pred = "다중밀집시설대형화재"
		elif y_pred[0] == 2 :
			pred = "철도교통사고"
		elif y_pred[0] == 3 :
			pred = "해양선박사고"
		elif y_pred[0] == 4 :
			pred = "항공기사고"
		elif y_pred[0] == 5 :
			pred = "도로교통사고"
		elif y_pred[0] == 6 :
			pred = "유해화학물질사고"
		elif y_pred[0] == 7 :
			pred = "감염병"

		score = confidence[0]
		Construction_score = round((score.tolist()[0])*100)
		Fire_score = round((score.tolist()[1])*100)
		train_score = round((score.tolist()[2])*100)
		ship_score = round((score.tolist()[3])*100)
		aircraft_score = round((score.tolist()[4])*100)
		road_score = round((score.tolist()[5])*100)
		chemical_score = round((score.tolist()[6])*100)
		infect_score = round((score.tolist()[7])*100)
  
		# # 임시저장 - 추론시 도출된 적당값 대치  
		# train_score = random.randrange(6,11)
		# ship_score = random.randrange(7,13)
		# aircraft_score = random.randrange(5,14)
		# road_score = random.randrange(8,15)
		# chemical_score = random.randrange(3,9)
		# infect_score = random.randrange(2,8)

		end = time.time()
		print("-----------------------------------------")
		print(">> 추론된 값은 ==> " , pred )
		print("> 사업장대규모인적사고의 confidence score :  " , Construction_score,"%" )
		print("> 다중밀집시설대형화재의 confidence score :  " , Fire_score,"%" )
		print("> 철도교통사고의 confidence score :  " , train_score,"%" )
		print("> 해양선박사고의 confidence score :  " , ship_score,"%" )
		print("> 항공기사고의 confidence score :  " , aircraft_score,"%" )
		print("> 도로교통사고의 confidence score :  " , road_score,"%" )
		print("> 유해화학물질사고의 confidence score :  " , chemical_score,"%" )
		print("> 감염병의 confidence score :  " , infect_score,"%" )


  
		log.write("-----------------------------------------\n")
		log.write(">> 추론된 값은 ==> {}\n".format(pred) )
		log.write("> 사업장대규모인적사고의 confidence score : {} % \n".format(Construction_score))
		log.write("> 다중밀집시설대형화재의 confidence score : {} % \n".format(Fire_score))
		log.write("> 철도교통사고의 confidence score : {} % \n".format(train_score))
		log.write("> 해양선박사고의 confidence score : {} % \n".format(ship_score))
		log.write("> 항공기사고의 confidence score : {} % \n".format(aircraft_score))
		log.write("> 도로교통사고의 confidence score : {} % \n".format(road_score))
		log.write("> 유해화학물질사고의 confidence score : {} % \n".format(chemical_score))
		log.write("> 감염병의 confidence score : {} % \n".format(infect_score))


                                  
		bardata = []
		bardata.append({"title":"사업장대규모인적사고"," value":Construction_score})
		bardata.append({"title":"다중밀집시설대형화재"," value":Fire_score})
		bardata.append({"title":"철도교통사고"," value":train_score})
		bardata.append({"title":"해양선박사고"," value":ship_score})
		bardata.append({"title":"항공기사고"," value":aircraft_score})
		bardata.append({"title":"도로교통사고"," value":road_score})
		bardata.append({"title":"유해화학물질사고"," value":chemical_score})
		bardata.append({"title":"감염병"," value":infect_score})


		print("----- json out -----")
		log.write("----- json out -----\n")
		print(bardata)
		for i in bardata :
			log.write(str(i)+"\n")
		print('[predict Complete: {}]'.format(datetime.timedelta(seconds=end-start)))
		log.write('[predict Complete: {}]'.format(datetime.timedelta(seconds=end-start)))

	with open("./classification_log.txt", 'r', encoding='utf-8') as f:
		c_log = f.read()

	
	return c_log , pred , bardata # 로그, 스텝 , 추론값 , bar json
	


# if __name__=='__main__':

# # 	#Test
# 	file_name = str(sys.argv[1])
# 	class_log, pred , bardata = Classificaion_predict(file_name)