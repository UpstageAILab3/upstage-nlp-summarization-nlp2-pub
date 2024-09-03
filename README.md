# **💁🏻🗨️💁🏻‍♂️대화 요약 Baseline code**

## 개요

> - kimkihong / helpotcreator@gmail.com / Upstage AI Lab 3기
> - 2024.08.29.목 10:00 ~ 2024.09.10.화 19:00

## 파일 소개

- kkh-1-data-meta.py
    - meta.csv 파일 내용에 한국어 내용을 추가하여 meta_kr.csv 파일로 만들어 줌
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

## 우분투에 miniconda3 설치

- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
- chmod +x Miniconda3-latest-Linux-x86_64.sh
- ./Miniconda3-latest-Linux-x86_64.sh
- conda create -n nlp python=3.10
- conda init
- rm Miniconda3-latest-Linux-x86_64.sh
- source ~/.bashrc
- conda activate nlp
- pip install jupyter nbconvert numpy matplotlib seaborn scikit-learn timm torch==2.3.1 pyyaml tqdm torch pytorch-lightning rouge transformers transformers[torch] wandb datasets absl-py nltk rouge_score evaluate konlpy fastapi uvicorn

## miniconda3 세팅_우분투_bash(선택)
우분투 bash 쉘 시작할 때, nlp 가상환경이 기본으로 실행되도록 하는 방법임.

- vim ~/.bashrc
- 가장 아래에 다음 두 줄 추가
    - conda deactivate
    - conda activate nlp
- source ~/.bashrc

## miniconda3 세팅_윈도우_cmd(선택)
cmd 시작할 때, 어떤 가상환경도 실행되지 않도록 하는 방법임.

- conda config --set auto_activate_base false

## jupyter_to_python.sh 파일 작성(선택)

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