# NER Project

---

## 1. NER?

1. Named Entity Recognition

> 문자열 안에서 NE의 위치를 알아내고, 사전(事前)정의한 카테고리에 따라 알맞게 분류하는 작업
>
> "NER is the process of locating and classifying named entities in text into predefined entity categories."

![img](https://blog.kakaocdn.net/dn/drYciy/btqLFnDwVkK/CANCk9dbhG4si47ZmgKkqk/img.png)

1. **규칙 기반 접근 (Rule-based Approches)**
   : domain-specific한 사전(gazetteer)을 적용하거나 패턴을 적용해서 접근
   : 높은 정확도이지만 낮은 재현율을 보이며 특히 다른 도메인으로 가면 성능이 급감함

2. **비지도 학습 접근 (Unsupervised Learning Approches)**
   : 문맥적 유사도에 기반해 clustering 하는 식으로 학습
   : 사전(gazetteer)을 만드는 데에 unsupervised system을 제안됨. 이는 지도학습과 비교해, 용어집이나 코퍼스의 통계적 정보(idf나 context vector), 혹은 얕은 수준의 통사적(syntactic) 지식에 의존함

3. **변수 기반 지도 학습 접근 (Feature-based Supervised Learning Approches)**
   : 지도학습으로 넘어가면 NER은 multi-class classification이나 sequence labeling task 영억으로 넘어감.
   : 'feature-based'이기 때문에, 'feature'가 무엇이 될 것이냐가 핵심
   : Hidden Markov Models(HMM), Decision Trees, Maximum Entropy Models, Support Vector Machines(SVM), Conditional Random Fields(CRF)
   : SVM 모델은 entity label을 예측할 때 이웃 단어는 고려하지 않는 반면, CRF는 고려함

4. 딥러닝 기반 NER 모델

   ; ![img](https://blog.kakaocdn.net/dn/bfwfe2/btqLFoPVMvS/NqYjt1QNNKADk0IjVfnkeK/img.png)

   : 논문[[A Survey on Deep Learning for Named Entity Recognition](https://arxiv.org/abs/1812.09449)]에서 NER 작업의 모델 구조를 세단계의 프로세스로 나눠서 제시

   1. **Distributed Representations for Input**
      : Pre-trained word embedding, Character-level embedding, POS tag, Gazetteer,...
   2. **Context Encoder**
      : CNN, RNN, Language model, Transformer,...
   3. **Tag Decoder**
      : Softmax, CRF, RNN, Point network,...

   - 2020년 03월까지의 NER 모델의 성능비교표

     ![img](https://blog.kakaocdn.net/dn/AetZu/btqLOOe3Lx0/wq02zkKNxNo2VHjki8nUWk/img.png)



## 2. 한국어 NER 데이터셋

한국어 데이터셋은 모두 공개되어 있으나 **상업적 이용은 허락하지 않고 있으니 주의해서 사용해야함**

1. **국립국어원[[5](https://ithub.korean.go.kr/user/total/referenceView.do?boardSeq=5&articleSeq=118&boardGb=T&isInsUpd=&boardType=CORPUS)] (5개)**
   : 장소(LC), 날짜(DT), 기관(OG), 시간(TI), 인물(PS)

2. **부산해양대학교 자연어처리 연구소[[6](https://github.com/kmounlp/NER/blob/master/NER Guideline (ver 1.0).pdf)] (10개)**
   : 인물(PER), 기관(ORG), 지명(LOC), 기타(POH), 날짜(DAT), 시간(TIM), 기간(DUR), 통화(MNY), 비율(PNT), 기타 수량표현(NOH)

3. **Naver NER Challenge[[7](http://air.changwon.ac.kr/?page_id=10)]([dataset](https://github.com/naver/nlp-challenge/tree/master/missions/ner)) (14개)**
   : 인물(PER), 학문분야(FLD), 인공물(AFW), 기관 및 단체(ORG), 지역명(LOC), 문명 및 문화(CVL), 날짜(DAT), 시간(TIM), 숫자(NUM), 사건 사고 및 행사(EVT), 동물(ANM), 식물(PLT), 금속/암석/화학물질(MAT), 의학용어/IT관련 용어(TRM)

4. HLCT 2016에서 제공한 데이터 세트 원본의 일부 오류를 수정하고 공개한 말뭉치
   : [KoreanNERCorpus](https://github.com/machinereading/KoreanNERCorpus)

5. 한국어 개체명 정의 및 표지 표준화 기술보고서와 이를 기반으로 제작된 개체명 형태소 말뭉치
   : [mounlp_NER](https://github.com/kmounlp/NER)

6. NIADic 기반 세종 2007, 공개말뭉치를 활용한 NER 데이터
   : [songys-entity](https://github.com/songys/entity)

   

## 3. 한국어 NER 구현

- [KoBERT + CRF](https://github.com/eagle705/pytorch-bert-crf-ner)
- [KoBERT(SKT)](https://github.com/SKTBrain/KoBERT)
- [HanBERT-NER](https://github.com/monologg/HanBert-NER)
- [DistillKoBERT](https://github.com/monologg/DistilKoBERT)



## 4. Reference

- https://stellarway.tistory.com/29 ; NER 설명
- https://littlefoxdiary.tistory.com/81 ; 한국어 사전학습모델 정리
- https://github.com/songys/entity; NER설명, 한국어 오픈 데이터 정리 및 총합