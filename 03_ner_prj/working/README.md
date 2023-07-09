# Pytorch-BERT-CRF-NER requirements 설치

### 가상환경

가상 환경 기준은 python 3.7.13 기준

### MXNET

pip install mxnet==1.8.0 -f https://dist.mxnet.io/python

### Pytorch

cuda 11.1 기준
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

### JPype

https://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype 위 경로에서 JPype 설치 예 : JPype1‑1.1.2‑cp37‑cp37m‑win_amd64.whl (cp37 = 파이선 3.7, amd64 = 64bit)

설치된 폴더에서

pip install JPype1-1.1.2-cp37-cp37m-win_amd64.whl

### Kobert 설치

pip install -e "git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf"
