# 📜 문서 타입 분류 대회

## 개요

> - kimkihong / helpotcreator@gmail.com / Upstage AI Lab 3기
> - 2024.07.30.화 10:00 ~ 2024.08.11.일 19:00

## 파일 소개

- kkh-1-data-meta.py
    - meta.csv 파일 내용에 한국어 내용을 추가하여 meta_kr.csv 파일로 만들어 줌
- kkh-2-data-trainimage(1. train_kr).py
    - train.csv 파일 내용 중, ID 부분에 한국어 클래스를 붙여서, train_kr.csv 파일로 만들어 줌
    - 추가로, train 폴더 내부의 이미지 파일명에 한국어 클래스를 붙인 이미지들을 생성하여 담은, train_kr 폴더를 생성함
- kkh-2-data-trainimage(2. train_kr_class).py
    - train_kr 폴더 내부의 이미지들을 클래스별로 나누어 폴더링하고, 그 모든 내용을 담은, train_kr_class 폴더를 생성함
- kkh-3-data-preprocessing.py
    - (아직 개발중)
- kkh-4-eda.ipynb
    - eda 진행한 내용임
- kkh-5-augmentation(1. train_kr_aug)_*.ipynb
    - train_kr 이미지를 대상으로 augmentation 진행하여, train_kr_aug 폴더 및 이미지들 생성함
    - train_kr.csv 파일 내용에도 augmentation 내용을 추가하여, train_kr_aug.csv 생성함
- kkh-6-model_*.ipynb
    - 학습, 평가, 앙상블
- kkh-7-evaluation_ensemble.py
    - 여러 .pt 파일을 지정하면, 앙상블 하드 보팅하고, 결과를 분석해준다.
- kkh-7-evaluation_justone.py
    - 1개 .pt 파일을 지정하면, 결과를 분석해준다.
- kkh-8-analyze_conf_*.ipynb
    - Train 데이터 학습 후, 해당 Train 이미지를 직접 평가해서, confidence를 계산한다.
- kkh-8-analyze_pred.ipynb
    - 컨퓨전 매트릭스
- kkh-square.py
    - 원본 이미지의 좌우 또는 상하에 padding을 추가해서, 정사각형으로 만들어 준다.
- kkh-title.py
    - 원본 이미지의 상단 20%를 crop하여 제목 부분만 추출한다.
- kkh-util-disk.py
    - 서버 사용량 확인
- font/
    - 폰트 파일
- pyproject.toml
    - 프로젝트 패키지 관리를 위한 poetry 설정 파일
- jupyter_to_python.sh
    - 주피터 파일을 파이썬 파일로 변환하는 리눅스 스크립트


## 우분투에 git 세팅

- apt update
- apt install -y git wget htop curl vim libgl1-mesa-glx libglib2.0-0
- git --version
- git config --global user.email "helpotcreator@gmail.com"
- git config --global user.name "helpotcreator"
- cd /
- git clone https://{개인 토큰}@github.com/UpstageAILab3/upstage-nlp-summarization-nlp2.git
- mv upstage-nlp-summarization-nlp2 kkh
- cd kkh
- git remote -v
- git checkout -b kimkihong origin/kimkihong
- git branch -a

## data.tar.gz 세팅

- cd /kkh
- wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000320/data/data.tar.gz
- tar -xzvf data.tar.gz
- rm data.tar.gz

## 우분투에 miniconda3 세팅

- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
- chmod +x Miniconda3-latest-Linux-x86_64.sh
- ./Miniconda3-latest-Linux-x86_64.sh
- conda create -n nlp python=3.10
- conda init
- rm Miniconda3-latest-Linux-x86_64.sh
- source ~/.bashrc
- conda activate nlp
- pip install jupyter nbconvert numpy matplotlib seaborn scikit-learn timm torch==2.3.1 pyyaml tqdm torch pytorch-lightning rouge transformers transformers[torch] wandb

## jupyter_to_python.sh 파일 작성

```bash
#!/bin/bash

# 주피터 노트북 파일명을 인자로 받음
NOTEBOOK_FILE="$1"

# 파일명이 주어지지 않으면 에러 메시지를 출력하고 종료
if [ -z "$NOTEBOOK_FILE" ]; then
    echo "Usage: $0 <notebook-file>"
    exit 1
fi

# 주어진 파일이 .ipynb 확장자를 가지고 있는지 확인
if [[ "$NOTEBOOK_FILE" != *.ipynb ]]; then
    echo "Error: The input file must have a .ipynb extension"
    exit 1
fi

# jupyter nbconvert 명령어를 사용하여 노트북 파일을 Python 스크립트로 변환
python -m jupyter nbconvert --to script "$NOTEBOOK_FILE"

# 변환 결과 확인
if [ $? -eq 0 ]; then
    echo "Conversion successful: ${NOTEBOOK_FILE%.ipynb}.py"
else
    echo "Conversion failed"
    exit 1
fi
```

## jupyter_to_python.sh 파일 세팅

- chmod +x jupyter_to_python.sh
- poetry run ./jupyter_to_python.sh {주피터 파일명}.ipynb
- poetry run python {만들어진 파이썬 파일}.py