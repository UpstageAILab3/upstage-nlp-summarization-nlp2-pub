# 환경 세팅

### 1. 가상 환경 생성

    conda create --name basecode

### 2. 라이브러리 설치

    pip install -r requirements.txt

### 3. wandb 설정

    wandb init

# 사용법

## 1. 빠른 시작

    py main.py

## 2. 파라미터 세팅 변경

config.yaml 파일에서 세팅을 변경해서 원하는 대로사용.

변경 해야할 주요 파라미터

- path에 있는 경로들
- model > model_select 로 사용 모델 변경
- vald > batch_size & train > batch_size & gradient_accumulation_step : 작을 수록 학습은 느리지만 성능이 올라갈 수 있음(오버피팅 주의) + 작게 하면 cuda 메모리에 부담이 덜 함.
- train > epoch : 1로 설정되어 있어요!!
- train > run_name : wandb에 저장할 때 run 이름

## 3. 모델 성능 평가 streamlit 사용법

    streamlit run result_visualization.py

명령어 실행하면 streamlit 실행

valid에 나온 파일을 로컬로 다운 받아 streamlit에 전달.



