[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/hm5nZYSf)
# FastCampus AI Lab NLP 프로젝트 - 7조
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [장은혁](https://github.com/UpstageAILab)             |            [김채원](https://github.com/UpstageAILab)             |            [이재민](https://github.com/UpstageAILab)             |            [김승호](https://github.com/UpstageAILab)             |            [송현지](https://github.com/UpstageAILab)             |            [기원선](https://github.com/UpstageAILab)             |
|                            팀장, 데이터 증강, 모델 실험                             |                            모델 실험, 성능 개선                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

### Overview

- Dialogue Summarization | 일상 대화 요약
학교 생활, 직장, 치료, 쇼핑, 여가, 여행 등 광범위한 일상 생활 중 하는 대화들에 대해 요약

### Timeline

- March 08, 2024 - Start Date
- March 20, 2024 - Final submission deadline

### Evaluation

- 예측된 요약 문장을 3개의 정답 요약 문장과 비교하여 metric의 평균 점수를 산출합니다. 본 대회에서는 ROUGE-1-F1, ROUGE-2-F1, ROUGE-L-F1, 총 3가지 종류의 metric으로부터 산출된 평균 점수를 더하여 최종 점수를 계산

## 2. Components

### Directory

- final_code (EDA는 따로 진행하지 않아 없음)

e.g.
```
├── code
│   └── final_code.py
├── docs
│   ├── pdf
│   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_NLP .pptx
└── input
    └── data
        ├── eval
        ├── dev
        └── train
```

## 3. Data descrption

### Dataset overview

- 모든 데이터는 .csv 형식으로 제공되고 있으며, 데이터는 아래와 같은 형태이며, 최소2턴, 최대 60턴으로 대화가 구성
    - train : 12457
    - dev : 499
    - test : 250
    - hidden-test : 249

### process data and build dataset classes

- 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력을 생성
- BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행
- Train, validation, test에 사용되는 Dataset 클래스를 정의
- tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력


### Building Trainers and TrainingArguments

- 모델 성능에 대한 평가 지표를 정의 (ROUGE)
- 미리 정의된 불필요한 생성토큰들을 제거
- 학습을 위한 trainer 클래스와 매개변수를 정의

### Modeling Process

- device를 정의
- 사용할 모델과 tokenizer 및 데이터셋 불러오기
- Trainer 클래스를 불러와 모델 학습 진행

### Inferring models

- Tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력
- 추론을 위한 tokenizer와 학습시킨 모델을 불러오기
- 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거
- 학습된 모델의 test를 진행


## 5. Result

### Leader Board

- <img width="963" alt="image" src="https://github.com/UpstageAILab/upstage-nlp-summarization-nlp7/assets/144979109/29216a40-ee98-4765-8d4a-c43511eacc90">

- 5등 (39.6005 / 0.4993 / 0.2990 / 0.3967)

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Reference

- (https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments)
- https://dacon.io/competitions/official/235673/data
- https://github.com/hunminjeongeum-korean-competition-2021/dialogue-summarization?tab=readme-ov-file#%EB%AC%B8%EC%A0%9C-1-%EB%AC%B8%EC%84%9C%EC%9A%94%EC%95%BD-dataset-%EC%84%A4%EB%AA%85
- https://tech.scatterlab.co.kr/alaggung-dlaggung-dialog-summary/
